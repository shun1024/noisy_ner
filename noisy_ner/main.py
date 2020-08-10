import copy
import os
import pickle
import random
import tempfile

from absl import app
from absl import flags
from absl import logging

from flair.embeddings import CharacterEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

from google.cloud import storage

"""
from trainer import ModelTrainer
from embeddings import CaseEmbedding, BertEmbeddings
"""

from noisy_ner.trainer import ModelTrainer
from noisy_ner.embeddings import CaseEmbedding, BertEmbeddings

FLAGS = flags.FLAGS

# Data
flags.DEFINE_boolean('is_gcp', False, 'whether run on GCP')
flags.DEFINE_string('dataset', './data/test', 'dataset folder')
flags.DEFINE_string('teacher_dir', None, 'directory with teacher init ckpt and corpus')
flags.DEFINE_string('output_dir', './output', 'output directory')

# Model flags
flags.DEFINE_integer('number_rnn_layers', 2, 'number of rnn layers')
flags.DEFINE_integer('hidden_size', 128, 'number of hidden size')
flags.DEFINE_float('dropout', 0.1, 'overall dropout rate')
flags.DEFINE_float('word_dropout', 0.05, 'word dropout rate')
flags.DEFINE_float('locked_dropout', 0.5, 'dropout rate for whole embedding')

# Optimization flags
flags.DEFINE_integer('epoch', 50, 'number of epochs')
flags.DEFINE_enum('optimizer', 'SGD', ['SGD', 'Adam'], 'optimizer')
flags.DEFINE_float('learning_rate', 0.3, 'learning rate')
flags.DEFINE_integer('batch_size', 32, 'batch size')

# Unlabel flags
flags.DEFINE_float('training_ratio', 1, 'percentage of label data')
flags.DEFINE_integer('unlabel_batch_ratio', 0, 'unlabel batch size = ratio * batch size')
flags.DEFINE_float('unlabel_weight', 1, 'weight for unlabel loss')
flags.DEFINE_string('augmentation', 'word_replace', 'augmentation methods')
flags.DEFINE_float('augmentation_strength', 0.15, 'strength for augmentations')
flags.DEFINE_float('temperature', 1, 'temperature for teacher model')


def get_exp_name(names):
    exp_name = []
    for name in names:
        short_name = name[0] + name[-1]
        value = getattr(FLAGS, name)
        exp_name.append('{}={}'.format(short_name, value))
    return ','.join(exp_name)


def parsing_gcs_path(gcs_path):
    tmp = gcs_path.split('/')
    return tmp[0], '/'.join(tmp[1:])


def download_folder_from_gcs(local_folder, gcs_path):
    os.makedirs(local_folder, exist_ok=True)
    gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.get_bucket(gcs_bucket)
    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        local_path = os.path.join(local_folder, os.path.basename(blob.name))
        logging.info('downloading: {} to {}'.format(blob.path, local_path))
        blob.download_to_filename(local_path)


def upload_folder_to_gcs(local_folder, gcs_path):
    gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.get_bucket(gcs_bucket)
    for dir_path, _, filenames in os.walk(local_folder):
        for name in filenames:
            filename = os.path.join(dir_path, name)
            blob = storage.Blob(os.path.join(gcs_folder, name), bucket)
            with open(filename, 'rb') as f:
                blob.upload_from_file(f)
            logging.info('uploading: {} to {}'.format(filename, blob.path))


def prepare_temp_dir(dataset):
    # prepare input and output directory
    logging.info('Prepare local directory')
    temp_dir = tempfile.mkdtemp()
    temp_indir, temp_outdir = os.path.join(temp_dir, 'inputs'), os.path.join(temp_dir, 'outputs')
    os.makedirs(temp_indir, exist_ok=True)
    os.makedirs(temp_outdir, exist_ok=True)

    download_folder_from_gcs(temp_indir, dataset)
    if FLAGS.teacher_dir is not None:
        download_folder_from_gcs(temp_indir, FLAGS.teacher_dir)

    return temp_indir, temp_outdir


def load_dataset(dataset_folder):
    # load dataset
    columns = {0: 'text', 1: 'position', 2: 'start', 3: 'end', 4: 'ner'}
    corpus = ColumnCorpus(dataset_folder, columns,
                          train_file='train.txt',
                          test_file='test.txt',
                          dev_file='dev.txt')
    return corpus


def remove_labels(corpus, label_ratio):
    # split label/unlabel
    training_sentences = corpus.train.sentences
    random.shuffle(training_sentences)
    remain_num = int(len(training_sentences) * label_ratio)
    remain_sentences, removed_sentences = training_sentences[: remain_num], training_sentences[remain_num:]
    corpus.train.total_sentence_count = remain_num
    corpus.train.sentences = remain_sentences

    unlabel_data = copy.deepcopy(corpus.train)
    if label_ratio < 1:
        unlabel_data.total_sentence_count = len(removed_sentences)
        unlabel_data.sentences = removed_sentences
    return corpus, unlabel_data


def normalize_corpus(corpus, unlabel_data):
    # convert all digits to zeros
    def digit_to_zero(string):
        result = []
        for char in string:
            if char.isdigit():
                result.append('0')
            else:
                result.append(char)
        return ''.join(result)

    def normalize(dataset):
        for i in range(len(dataset.sentences)):
            sentence = Sentence()
            for token in dataset.sentences[i]:
                text = token.text
                text = digit_to_zero(text)
                new_token = copy.deepcopy(token)
                new_token.text = text
                sentence.add_token(new_token)
            dataset.sentences[i] = sentence
        return dataset

    for dataset in [corpus.train, corpus.dev, corpus.test, unlabel_data]:
        normalize(dataset)
    return corpus, unlabel_data


def init_from_ckpt(temp_indir, tagger):
    logging.info('Loading teacher ckpt from: {}'.format(temp_indir))
    model_path = os.path.join(temp_indir, 'final.ckpt')
    corpus_path = os.path.join(temp_indir, 'corpus.pickle')
    tagger = tagger.load(model_path)
    corpus, unlabel_data = pickle.load(open(corpus_path, 'rb'))
    logging.info('Completed loading !!!')
    return tagger, corpus, unlabel_data


def save_to_ckpt(temp_outdir, tagger, corpus, unlabel_data):
    # save model with corpus to local directory
    last_model_path = os.path.join(temp_outdir, 'final.ckpt')
    corpus_path = os.path.join(temp_outdir, 'corpus.pickle')
    tagger.save(last_model_path)
    pickle.dump((corpus, unlabel_data), open(corpus_path, 'wb'))


def main(_):
    exp_name = get_exp_name(['training_ratio', 'epoch', 'hidden_size', 'dropout', 'learning_rate'])
    logging.info('Start Exp: {}'.format(exp_name))

    if FLAGS.is_gcp:
        temp_indir, temp_outdir = prepare_temp_dir(FLAGS.dataset)
    else:
        temp_indir, temp_outdir = FLAGS.dataset, os.path.join(FLAGS.output_dir, exp_name)

    corpus = load_dataset(temp_indir)
    corpus, unlabel_data = remove_labels(corpus, FLAGS.training_ratio)
    corpus, unlabel_data = normalize_corpus(corpus, unlabel_data)
    tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
    embeddings = StackedEmbeddings(embeddings=[FlairEmbeddings('news-forward'),
                                               CharacterEmbeddings(),
                                               CaseEmbedding()])

    tagger = SequenceTagger(hidden_size=FLAGS.hidden_size,
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
    trainer.train(temp_outdir,
                  unlabel_data=unlabel_data,
                  unlabel_batch_ratio=FLAGS.unlabel_batch_ratio,
                  unlabel_weight=FLAGS.unlabel_weight,
                  augment_prob=FLAGS.augmentation_strength,
                  augment_method=FLAGS.augmentation,
                  temperature=FLAGS.temperature,
                  learning_rate=FLAGS.learning_rate,
                  mini_batch_size=FLAGS.batch_size,
                  max_epochs=FLAGS.epoch,
                  embeddings_storage_mode='none',
                  checkpoint=False,
                  save_final_model=False)

    save_to_ckpt(temp_outdir, tagger, corpus, unlabel_data)

    if FLAGS.is_gcp:
        gcs_path = os.path.join(FLAGS.output_dir, exp_name)
        upload_folder_to_gcs(temp_outdir, gcs_path)
    logging.info('Finished !!!')


if __name__ == '__main__':
    app.run(main)
