import os, glob, shutil

from absl import app
from absl import flags
from absl import logging

import flair
from flair.datasets import ColumnCorpus

from noisy_ner.utils import prepare_temp_dir
from noisy_ner.utils import remove_labels, normalize_corpus
from noisy_ner.utils import init_from_ckpt
from noisy_ner.trainer import ModelTrainer
from noisy_ner.tagger import CustomTagger
from noisy_ner.embeddings import get_embedding

FLAGS = flags.FLAGS

# Data
flags.DEFINE_boolean('is_gcp', False, 'whether run on GCP')
flags.DEFINE_string('dataset', './data/test', 'dataset folder')
flags.DEFINE_string('out_dataset', None, 'out domain dataset')
flags.DEFINE_string('teacher_dir', None, 'directory with teacher init ckpt and corpus')
flags.DEFINE_string('output_dir', './output', 'output directory')
flags.DEFINE_string('exp', 'test', 'output directory')
flags.DEFINE_string('embedding', 'bert', 'embedding type')
flags.DEFINE_boolean('train_from_scratch', False, 'whether to train student from scratch')
flags.DEFINE_boolean('finetune_bert', False, 'whether to adjust bert')
flags.DEFINE_integer('train_with_dev', 1, 'train with dev')

# Model flags
flags.DEFINE_integer('number_rnn_layers', 2, 'number of rnn layers')
flags.DEFINE_integer('hidden_size', 256, 'number of hidden size')
flags.DEFINE_float('dropout', 0.2, 'overall dropout rate')
flags.DEFINE_float('word_dropout', 0.05, 'word dropout rate')
flags.DEFINE_float('locked_dropout', 0.5, 'dropout rate for whole embedding')

# Optimization flags
flags.DEFINE_integer('epoch', 100, 'number of epochs')
flags.DEFINE_enum('optimizer', 'SGD', ['SGD', 'Adam'], 'optimizer')
flags.DEFINE_float('learning_rate', 0.3, 'learning rate')
flags.DEFINE_integer('batch_size', 32, 'batch size')

# Unlabel flags
flags.DEFINE_float('training_ratio', 1, 'percentage of label data')
flags.DEFINE_integer('unlabel_batch_ratio', 0, 'unlabel batch size = ratio * batch size')
flags.DEFINE_boolean('update_teacher', False, 'whether to update teacher')
flags.DEFINE_float('unlabel_weight', 1, 'weight for unlabel loss')
flags.DEFINE_string('augmentation', 'word_replace', 'augmentation methods')
flags.DEFINE_float('augmentation_strength', 0.15, 'strength for augmentations')
flags.DEFINE_float('temperature', 1, 'temperature for teacher model')


def load_dataset(dataset_folder):
    if 'conll_03' in dataset_folder:
        temp_conll_dir = os.path.join(dataset_folder, 'conll_03')
        os.makedirs(temp_conll_dir, exist_ok=True)
        for filename in glob.glob(os.path.join(dataset_folder, 'eng*')):
            shutil.copy(filename, temp_conll_dir)
        corpus = flair.datasets.sequence_labeling.CONLL_03(dataset_folder)
    else:
        columns = {0: 'text', 1: 'position', 2: 'start', 3: 'end', 4: 'ner'}
        corpus = ColumnCorpus(dataset_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
    return corpus


def main(_):
    exp_name = FLAGS.exp
    logging.info('Start Exp: {}'.format(exp_name))

    if FLAGS.is_gcp:
        temp_indir, temp_outdir = prepare_temp_dir(FLAGS.dataset, FLAGS.teacher_dir)
    else:
        temp_indir, temp_outdir = FLAGS.dataset, os.path.join(FLAGS.output_dir, exp_name)

    corpus = load_dataset(temp_indir)

    out_corpus = None
    if FLAGS.out_dataset is not None:
        if FLAGS.is_gcp:
            temp_out_indir, _ = prepare_temp_dir(FLAGS.out_dataset, FLAGS.teacher_dir)
        else:
            temp_out_indir = FLAGS.out_dataset
        out_corpus = load_dataset(temp_out_indir)

    if FLAGS.train_with_dev:
        corpus.train.sentences = corpus.train.sentences + corpus.dev.sentences
        corpus.train.total_sentence_count = len(corpus.train.sentences)
    # TODO (shunl): hack for monitoring, to be removed
    corpus = flair.data.Corpus(corpus.train, corpus.test, corpus.test, name='dataset')

    corpus, unlabel_data = remove_labels(corpus, FLAGS.training_ratio)
    corpus, unlabel_data = normalize_corpus(corpus, unlabel_data)
    tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
    embeddings = get_embedding(FLAGS.embedding, finetune_bert=FLAGS.finetune_bert)
    tagger = CustomTagger(hidden_size=FLAGS.hidden_size,
                          dropout=FLAGS.dropout,
                          word_dropout=FLAGS.word_dropout,
                          locked_dropout=FLAGS.locked_dropout,
                          rnn_layers=FLAGS.number_rnn_layers,
                          embeddings=embeddings,
                          tag_dictionary=tag_dictionary,
                          tag_type='ner',
                          use_crf=True)

    if FLAGS.teacher_dir is not None:
        if FLAGS.is_gcp:
            FLAGS.teacher_dir = temp_indir
        tagger, corpus, unlabel_data = init_from_ckpt(FLAGS.teacher_dir, tagger)

    trainer = ModelTrainer(tagger, corpus, use_tensorboard=True)
    train_step_ratio = min(15, max(3, int(1 / FLAGS.training_ratio)))

    if out_corpus is not None:
        # add training as unlabel
        unlabel_data.total_sentence_count += len(out_corpus.train.sentences)
        unlabel_data.sentences += out_corpus.train.sentences

    trainer.train(temp_outdir,
                  is_gcp=FLAGS.is_gcp,
                  gcp_dir=os.path.join(FLAGS.output_dir, exp_name),
                  unlabel_data=unlabel_data,
                  out_corpus=out_corpus,
                  unlabel_batch_ratio=FLAGS.unlabel_batch_ratio,
                  unlabel_weight=FLAGS.unlabel_weight,
                  augment_prob=FLAGS.augmentation_strength,
                  augment_method=FLAGS.augmentation,
                  temperature=FLAGS.temperature,
                  train_step_ratio=train_step_ratio,
                  learning_rate=FLAGS.learning_rate,
                  train_from_scratch=FLAGS.train_from_scratch,
                  mini_batch_size=FLAGS.batch_size,
                  max_epochs=FLAGS.epoch,
                  update_teacher=FLAGS.update_teacher,
                  embeddings_storage_mode='none')

    logging.info('Finished !!!')


if __name__ == '__main__':
    app.run(main)
