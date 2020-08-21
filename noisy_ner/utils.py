import tempfile
import pickle
import copy, os, random

from absl import logging

from flair.data import Sentence

from google.cloud import storage


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


def prepare_temp_dir(dataset, teacher_dir=None):
    # prepare input and output directory
    logging.info('Prepare local directory')
    temp_dir = tempfile.mkdtemp()
    temp_indir, temp_outdir = os.path.join(temp_dir, 'inputs'), os.path.join(temp_dir, 'outputs')
    os.makedirs(temp_indir, exist_ok=True)
    os.makedirs(temp_outdir, exist_ok=True)

    download_folder_from_gcs(temp_indir, dataset)
    download_folder_from_gcs('./glove', 'deid-xcloud/data/glove')
    if teacher_dir is not None:
        download_folder_from_gcs(temp_indir, teacher_dir)

    return temp_indir, temp_outdir


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
    logging.info('Saving teacher ckpt')
    last_model_path = os.path.join(temp_outdir, 'final.ckpt')
    corpus_path = os.path.join(temp_outdir, 'corpus.pickle')
    tagger.save(last_model_path)
    pickle.dump((corpus, unlabel_data), open(corpus_path, 'wb'))
