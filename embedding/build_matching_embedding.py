import os, sys, glob
from gensim.models import KeyedVectors, Word2Vec


def read_all_sentences(infile):
    results = []

    tmp = []
    with open(infile, 'w') as f:
        for line in f:
            if len(line.strip()) == 0:
                results.append(tmp)
                tmp = []
            else:
                word = line.split()[0].lower()
                tmp.append(word)
    return results


if __name__ == "__main__":
    data_dir = sys.argv[1]
    files = glob.glob(os.path.join(data_dir, '*'))
    sentences = []
    for infile in files:
        if 'glove.bin' in infile:
            continue

        sentences += read_all_sentences(infile)

    model = Word2Vec(sentences, size=300, min_count=1)
    model.build_vocab(sentences)
    training_examples_count = model.corpus_count

    glove_bin = '/h/sliao3/deid/noisy_ner/data/glove/glove.bin'
    glove = KeyedVectors.load()
    model.build_vocab([list(glove.keys())], update=True)
    model.intersect_word2vec_format(glove_bin, binary=True, lockf=1.0)
    model.train(sentences, total_examples=training_examples_count, epochs=model.iter)
    model.save(os.path.join(data_dir, 'glove.bin'))


