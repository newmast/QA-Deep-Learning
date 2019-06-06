import numpy as np
import constants

from io import open
from tensorflow.python.lib.io import file_io

def load_glove(stream, vocab=None):
    print('Loading GloVe vectors ..')

    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split(' ')) - 1
    lookup = np.empty([500000, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(' ', 1)[1], sep=' ')
    word2idx[first_line.split(' ', 1)[0]] = 0
    n = 1
    for line in stream:
        word, vec = line.rstrip().split(' ', 1)
        if vocab is None or word in vocab and word not in word2idx:
            idx = len(word2idx)
            word2idx[word] = idx
            if idx > np.size(lookup, axis=0) - 1:
                lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
            lookup[idx] = np.fromstring(vec, sep=' ')
        n += 1
    lookup.resize([len(word2idx), dim])
    print('Loading GloVe vectors completed.')
    return word2idx, lookup

if constants.IS_LOCAL:
    embs = open("glove.6B/glove.6B.100d.txt", "r", encoding='utf-8')
else:
    embs = file_io.FileIO('gs://squad-bucket/glove.6B', mode='r')

word2idx, lookup = load_glove(embs)

def get_word_index(word):
    try:
        return word2idx[word]
    except KeyError:
        return constants.VOCAB_SIZE

def get_embedding_matrix():
    return lookup
