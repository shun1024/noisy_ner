from pathlib import Path
from typing import Union, List
import copy, os, pickle, time, logging

import flair
import flair.nn
from flair.data import MultiCorpus, Corpus, Token
from flair.datasets import DataLoader
from flair.training_utils import (
    log_line,
    add_file_handler,
    Result,
    Metric,
    store_embeddings,
)
from flair.models import SequenceTagger

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter

from noisy_ner.augmentations import augment
from noisy_ner.utils import upload_folder_to_gcs

log = logging.getLogger("flair")


def save_to_ckpt(temp_outdir, tagger, corpus, unlabel_data):
    # save model with corpus to local directory
    last_model_path = os.path.join(temp_outdir, 'final.ckpt')
    corpus_path = os.path.join(temp_outdir, 'corpus.pickle')
    tagger.save(last_model_path)
    pickle.dump((corpus, unlabel_data), open(corpus_path, 'wb'))


def evaluate(
    model,
    data_loader: DataLoader,
    out_path: Path = None,
    embedding_storage_mode: str = "none",
) -> (Result, float):
    if type(out_path) == str:
        out_path = Path(out_path)

    with torch.no_grad():
        eval_loss = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=model.beta)

        lines: List[str] = []

        if model.use_crf:
            transitions = model.transitions.detach().cpu().numpy()
        else:
            transitions = None

        for batch in data_loader:
            batch_no += 1

            with torch.no_grad():
                features = model.forward(batch)
                loss = model._calculate_loss(features, batch)
                tags, _ = model._obtain_labels(
                    feature=features,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=False,
                )

            eval_loss += loss

            for (sentence, sent_tags) in zip(batch, tags):
                for (token, tag) in zip(sentence.tokens, sent_tags):
                    token: Token = token
                    token.add_tag("predicted", tag.value, tag.score)

                    # append both to file for evaluation
                    eval_line = "{} {} {} {}\n".format(
                        token.text,
                        token.get_tag(model.tag_type).value,
                        tag.value,
                        tag.score,
                    )
                    lines.append(eval_line)
                lines.append("\n")

            def add_tags(spans, tag_names, new_name):
                new_tags = []
                for tag in spans:
                    if tag.tag in tag_names:
                        new_tags.append([new_name, tag.text])
                return new_tags

            for sentence in batch:
                # make list of gold tags
                gold_tags = [
                    (tag.tag, tag.text) for tag in sentence.get_spans(model.tag_type)
                ]

                gold_tags += add_tags(sentence.get_spans(model.tag_type), ['PATIENT', 'DOCTOR'], 'NAME')

                # make list of predicted tags
                predicted_tags = [
                    (tag.tag, tag.text) for tag in sentence.get_spans("predicted")
                ]

                predicted_tags += add_tags(sentence.get_spans("predicted"), ['PATIENT', 'DOCTOR'], 'NAME')

                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)
                    else:
                        metric.add_tn(tag)

            store_embeddings(batch, embedding_storage_mode)

        eval_loss /= batch_no

        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        detailed_result = (
            f"\nMICRO_AVG: acc {metric.micro_avg_accuracy():.4f} - f1-score {metric.micro_avg_f_score():.4f}"
            f"\nMACRO_AVG: acc {metric.macro_avg_accuracy():.4f} - f1-score {metric.macro_avg_f_score():.4f}"
        )

        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss, metric.f_score('NAME')


class ModelTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        use_tensorboard: bool = True,
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
        :param use_tensorboard: If True, writes out tensorboard information
        """
        self.model: flair.nn.Model = model

        # init teacher / disable the gradient
        self.teacher: flair.nn.Model = copy.deepcopy(model)
        self.teacher.eval()

        self.best_model: flair.nn.Model = copy.deepcopy(model)
        self.corpus: Corpus = corpus
        self.optimizer: torch.optim.Optimizer = optimizer
        self.epoch: int = epoch
        self.use_tensorboard: bool = use_tensorboard

    def train(
        self,
        base_path: Union[Path, str],
        unlabel_data: flair.datasets.ColumnDataset,
        out_corpus: flair.data.Corpus = None,
        is_gcp: bool = False,
        gcp_dir: str = "",
        unlabel_batch_ratio: int = 0,
        unlabel_weight: float = 0,
        augment_method: str = 'word_replace',
        augment_prob: float = 0.15,
        temperature: float = 1,
        update_teacher: bool = False,
        train_from_scratch: bool = False,
        saving_fqs: int = 1,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 100,
        anneal_factor: float = 0.5,
        train_step_ratio: int = 5,
        patience: int = 2,
        min_learning_rate: float = 0.0001,
        embeddings_storage_mode: str = "cpu",
        shuffle: bool = True,
        num_workers: int = 6,
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param shuffle: If True, data is shuffled during training
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        if 0. the evaluation is not performed on fraction of training data,
        if 'dev' the size is determined from dev set size
        and kept fixed during training, otherwise it's sampled at beginning of each epoch
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        if train_from_scratch:
            log.info('Resetting the model')
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        if self.use_tensorboard:
            writer = SummaryWriter(base_path)

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")
        if isinstance(self.model, SequenceTagger) and self.model.weight_dict and self.model.use_crf:
            log_line(log)
            log.warning(f'WARNING: Specified class weights will not take effect when using CRF')

        optimizer: torch.optim.Optimizer = self.optimizer(
            self.model.parameters(), lr=learning_rate, **kwargs
        )

        anneal_mode = "max"
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer,
            factor=anneal_factor,
            patience=patience,
            mode=anneal_mode,
            verbose=True,
        )

        train_data = self.corpus.train

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            best_score = 0
            self.epoch = 0
            while self.epoch < max_epochs:
                log_line(log)

                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < min_learning_rate:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                )

                if unlabel_batch_ratio > 0:
                    unlabel_batch_loader = DataLoader(
                        unlabel_data,
                        batch_size=mini_batch_size * unlabel_batch_ratio,
                        shuffle=shuffle,
                        num_workers=num_workers,
                    )
                    token_list = []
                    print('reading all tokens')
                    for sentence in unlabel_data:
                        for token in sentence:
                            token_list.append(token.text)
                    token_list = list(set(token_list))
                    print('finished reading all tokens')

                def train_step():
                    self.model.train()
                    train_loss: float = 0
                    unsup_train_loss: float = 0

                    seen_batches = 0
                    total_number_of_batches = len(batch_loader)

                    modulo = max(1, int(total_number_of_batches / 10))

                    # process mini-batches
                    batch_time = 0
                    for batch_no, batch in enumerate(batch_loader):
                        start_time = time.time()

                        # zero the gradients on the model and optimizer
                        torch.cuda.empty_cache()
                        self.model.zero_grad()
                        optimizer.zero_grad()

                        # forward pass
                        loss = self.model._calculate_loss(self.model.forward(batch), batch)
                        loss.backward()

                        # teacher loss
                        if unlabel_batch_ratio > 0:
                            # calculate the unlabel loss
                            # if necessary, make batch_steps
                            unbatch_ori_batch = next(iter(unlabel_batch_loader))
                            teacher_output = self.teacher.forward(unbatch_ori_batch).detach()
                            teacher_prob = torch.nn.functional.softmax(teacher_output / temperature, -1)

                            unbatch_aug_batch = augment(unbatch_ori_batch, token_list, augment_method, augment_prob)
                            student_output = self.model.forward(unbatch_aug_batch)

                            # forward pass
                            unlabel_loss = unlabel_weight * torch.mean(
                                torch.sum(- teacher_prob * torch.nn.functional.log_softmax(student_output, -1), -1))
                            unlabel_loss.backward()

                        # do the optimizer step
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        optimizer.step()

                        seen_batches += 1
                        train_loss += loss.item()
                        if unlabel_batch_ratio > 0:
                            unsup_train_loss += unlabel_loss.item() * unlabel_weight
                            store_embeddings(unbatch_ori_batch, embeddings_storage_mode)

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(batch, embeddings_storage_mode)
                        batch_time += time.time() - start_time
                        if seen_batches % modulo == 0:
                            log.info(
                                f"epoch {self.epoch} - iter {seen_batches}/{total_number_of_batches} - loss "
                                f"{train_loss / seen_batches:.4f} - unlabel_loss {unsup_train_loss / seen_batches:.4f}"
                                f" - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                            )
                            batch_time = 0

                    train_loss /= seen_batches
                    if unlabel_batch_ratio > 0:
                        unsup_train_loss /= seen_batches

                    log_line(log)
                    log.info(
                        f"EPOCH {self.epoch} done: loss {train_loss:.4f} - unsup {unsup_train_loss:.4f} - lr {learning_rate:.4f}"
                    )

                    if self.use_tensorboard:
                        writer.add_scalar("train/loss", train_loss, self.epoch)
                        writer.add_scalar("train/learning_rate", learning_rate, self.epoch)
                        if unlabel_batch_ratio > 0:
                            writer.add_scalar("train/unsup_train_loss", unsup_train_loss, self.epoch)

                def dev_step(dev_corpus, dev_name):
                    # evaluate on train / dev / test split depending on training settings
                    result_line: str = ""
                    self.model.eval()
                    dev_eval_result, dev_loss, name_f1 = evaluate(
                        self.model,
                        DataLoader(
                            dev_corpus,
                            batch_size=mini_batch_size,
                            num_workers=num_workers,
                        ),
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
                    )

                    current_score = dev_eval_result.main_score

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(dev_corpus, embeddings_storage_mode)
                    if self.use_tensorboard:
                        writer.add_scalar(
                            "%s/micro_f1" % dev_name, dev_eval_result.main_score, self.epoch
                        )
                        writer.add_scalar(
                            "%s/name_micro_f1" % dev_name, name_f1, self.epoch
                        )
                        writer.add_scalar("%s/loss" % dev_name, dev_loss, self.epoch)

                    if self.epoch % saving_fqs == 0:
                        log.info("Saving model & corpus to local directory")
                        save_to_ckpt(base_path, self.model, self.corpus, unlabel_data)

                        if is_gcp:
                            log.info("Uploading model to cloud bucket")
                            upload_folder_to_gcs(base_path, gcp_dir)

                    return current_score

                current_score = dev_step(self.corpus.dev, 'dev')

                if out_corpus is not None:
                    dev_step(out_corpus.dev, 'out_dev')

                for i in range(train_step_ratio):
                    train_step()
                    self.epoch += 1

                # determine learning rate annealing through scheduler
                scheduler.step(current_score)

                # determine bad epoch number
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if new_learning_rate != previous_learning_rate:
                    bad_epochs = patience + 1

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # if we use dev data, remember best model based on dev evaluation score
                if current_score > best_score:
                    self.model.eval()
                    best_score = current_score
                    self.best_model = pickle.loads(pickle.dumps(self.model))
                    if update_teacher:
                        log.info(f"UPDATE TEACHER")
                        self.teacher = pickle.loads(pickle.dumps(self.model))
                        self.teacher.eval()

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

        # test best model if test data is present
        final_score = self.final_test(base_path, mini_batch_size, num_workers)
        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.add_scalar("test/score", final_score)
            writer.close()

        return {
            "test_score": final_score
        }

    def save_checkpoint(self, model_file: Union[str, Path]):
        corpus = self.corpus
        self.corpus = None
        torch.save(self, str(model_file), pickle_protocol=4)
        self.corpus = corpus

    @classmethod
    def load_checkpoint(cls, checkpoint: Union[Path, str], corpus: Corpus):
        model: ModelTrainer = torch.load(checkpoint, map_location=flair.device)
        model.corpus = corpus
        return model

    def final_test(
        self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8
    ):
        log_line(log)
        log.info("Testing using best model ...")

        self.best_model.eval()

        test_results, test_loss, _ = evaluate(
            self.best_model,
            DataLoader(
                self.corpus.test,
                batch_size=eval_mini_batch_size,
                num_workers=num_workers,
            ),
            out_path=base_path / "test.tsv",
            embedding_storage_mode="none",
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                evaluate(
                    self.best_model,
                    DataLoader(
                        subcorpus.test,
                        batch_size=eval_mini_batch_size,
                        num_workers=num_workers,
                    ),
                    out_path=base_path / f"{subcorpus.name}-test.tsv",
                    embedding_storage_mode="none",
                )

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score
