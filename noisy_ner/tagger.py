from pathlib import Path
from typing import List

import torch
import torch.nn
from torch.utils.data import DataLoader

from flair.data import Token
from flair.models import SequenceTagger
from flair.training_utils import Metric, store_embeddings


class Result(object):
    def __init__(
        self, main_score: float, log_header: str, log_line: str, detailed_results: str, sub_scores: List = None
    ):
        self.main_score: float = main_score
        self.sub_scores: List = sub_scores
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


class CustomTagger(SequenceTagger):

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation", beta=self.beta)

            lines: List[str] = []

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(
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
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")

                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, tag.text) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, tag.text) for tag in sentence.get_spans("predicted")
                    ]

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

                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        ('PHI', tag.text) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        ('PHI', tag.text) for tag in sentence.get_spans("predicted")
                    ]

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
                sub_scores=[metric.f_score('PHI'), metric.precision('PHI'), metric.recall('PHI')],
                log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss
