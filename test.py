from __future__ import division
# paragraph input - 1 x 650 x 100
# question input - 1 x 50 x 100

import numpy as np

#p: hey what up

#q: hey up
# paragraphs = np.array([
#     [
#         [ 1, 2, 3, 4, 5 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ]
#     ],
#     [
#         [ 1, 2, 3, 4, 5 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 1, 6, 4, 2 ]
#     ]
# ])
#
# questions = np.array([
#     [
#         [ 1, 2, 3, 4, 5 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 0, 0, 0, 0, 0 ],
#         [ 0, 0, 0, 0, 0 ]
#     ],
#     [
#         [ 1, 2, 3, 4, 5 ],
#         [ 0, 1, 6, 4, 2 ],
#         [ 6, 6, 4, 3, 2 ],
#         [ 0, 0, 0, 0, 0 ]
#     ]
# ])
#
# from keras import backend as K
#
# def wiq_b(P, Q):
#     wiq_b = np.zeros((P.shape[0], P.shape[1]))
#     for i in range(Q.shape[1]):
#         question_word = K.tf.expand_dims(Q[:, i, :], 1)
#         same_vector_components = K.tf.cast(K.tf.equal(question_word, P), K.tf.int32)
#         same_words_weights = K.tf.reduce_sum(same_vector_components, axis=2)
#         wiq_b = K.tf.logical_or(wiq_b, K.tf.equal(same_words_weights, P.shape[2]))
#
#     return K.tf.cast(wiq_b, K.tf.int32)
#
# def wiq_w(P, Q):
#     wiq_w = np.zeros((P.shape[0], P.shape[1]))
#     for i in range(Q.shape[1]):
#         question_word = K.tf.expand_dims(Q[:, i, :], 1)
#         hadamard_product_paragraphs = K.tf.cast(K.tf.reduce_sum(K.tf.multiply(P, question_word), axis=2), K.tf.float32)
#         softmaxed_product = K.tf.nn.softmax(hadamard_product_paragraphs)
#         wiq_w += softmaxed_product
#     return wiq_w
#
# def start_feature(P, Q):
#     K.tf.gather_nd(
#         arg[0],
#         K.tf.stack([
#             K.tf.range(K.tf.shape(arg[1])[0]),
#             K.tf.cast(arg[1], K.tf.int32)
#         ], axis=1)
#     )
# sess = K.tf.Session()
# with sess.as_default():
#     print(wiq_b(questions, questions).eval())

from training import glove_embeddings
import json
from training import evaluate
from training import constants
from training import preprocessing
import re
import numpy as np
from io import open

inverted = {v: k for k, v in glove_embeddings.word2idx.iteritems()}

starts = json.loads(open("starts.json", mode='r', encoding='utf-8').read())
ends = json.loads(open("ends.json", mode='r', encoding='utf-8').read())

p, q, anss, anse = preprocessing.load_preprocessed_dataset('processed-dev-v1.1.json')

anss = anss[:10000]
anse = anse[:10000]

wrongs = 0
f1 = 0
em = 0
for i in range(10000):
    index_answer_pred = p[i][starts[i]:ends[i]+1]
    index_answer_true = p[i][anss[i]:anse[i]+1]
    try:
        prediction = [inverted[index] for index in index_answer_pred]
        truth = [inverted[index] for index in index_answer_true]
        f1 += evaluate.f1_score(" ".join(prediction), " ".join(truth))
        em += evaluate.exact_match_score(" ".join(prediction), " ".join(truth))
    except KeyError:
        wrongs += 1
        continue

print("F1: " + str(f1 / (10000-wrongs)))
print("EM: " + str(em / (10000-wrongs)))
print("Wrongs: " + str(wrongs))
