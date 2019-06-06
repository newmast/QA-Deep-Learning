from __future__ import division

import numpy as np
import constants
import glove_embeddings

from tensorflow.python.lib.io import file_io
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Embedding, Input, Dense, RepeatVector, Masking, Lambda, Softmax, Dropout, Flatten, Activation, Reshape, Permute, merge, Multiply, Concatenate
from keras.layers.recurrent import LSTM, GRU

def wiq_b(P, Q):
    wiq_b = False
    for i in range(Q.shape[1]):
        question_word = K.tf.expand_dims(Q[:, i, :], 1)
        same_vector_components = K.tf.cast(K.tf.equal(question_word, P), K.tf.int32)
        same_words_weights = K.tf.reduce_sum(same_vector_components, axis=2)
        wiq_b = K.tf.logical_or(wiq_b, K.tf.equal(same_words_weights, P.shape[2]))
    return K.tf.cast(wiq_b, K.tf.float32)

def wiq_w(P, Q):
    wiq_w = 0
    for i in range(Q.shape[1]):
        question_word = K.tf.expand_dims(Q[:, i, :], 1)
        dot_product_paragraphs = K.tf.cast(K.tf.reduce_sum(K.tf.multiply(P, question_word), axis=2), K.tf.float32)
        softmaxed_product = K.tf.nn.softmax(dot_product_paragraphs)
        wiq_w += softmaxed_product
    return wiq_w

def get_question_wiq(Q):
    shape = K.tf.shape(Q)
    return K.tf.ones([shape[0], constants.MAX_QUESTION_LENGTH, 1], K.tf.float32)

embedding_matrix = glove_embeddings.get_embedding_matrix()
embedding_matrix = np.concatenate([embedding_matrix, np.zeros([1, constants.EMBEDDING_DIM])], axis=0)

paragraph_indices = Input(shape=(constants.MAX_PARAGRAPH_LENGTH, ), name='paragraph_as_word_indices')
question_indices = Input(shape=(constants.MAX_QUESTION_LENGTH, ), name='question_as_indices')

paragraph_input = Embedding(
    constants.VOCAB_SIZE + 1,
    constants.EMBEDDING_DIM,
    name="paragraph_embedding",
    input_length=constants.MAX_PARAGRAPH_LENGTH,
    weights=[embedding_matrix],
    trainable=False)(paragraph_indices)

question_input = Embedding(
    constants.VOCAB_SIZE + 1,
    constants.EMBEDDING_DIM,
    name="question_embedding",
    input_length=constants.MAX_QUESTION_LENGTH,
    weights=[embedding_matrix],
    trainable=False)(question_indices)

paragraph_wiq_b = Lambda(lambda arg: K.tf.expand_dims(wiq_b(arg[0], arg[1]), axis=2), name="paragraph_wiq_b")([paragraph_input, question_input])
paragraph_wiq_w = Lambda(lambda arg: K.tf.expand_dims(wiq_w(arg[0], arg[1]), axis=2), name="paragraph_wiq_w")([paragraph_input, question_input])

question_wiq = Lambda(lambda arg: get_question_wiq(arg[0]), name="question_wiq")([question_input])

paragraph_input_with_wiq = Concatenate(name="merge_paragraph_with_wiq")([
    paragraph_input,
    paragraph_wiq_b,
    paragraph_wiq_w
])

question_input_with_wiq = Concatenate(name="merge_question_with_wiq")([
    question_input,
    question_wiq,
    question_wiq
])

encoder = Bidirectional(LSTM(units=constants.EMBEDDING_DIM,
                         return_sequences=True,
                         dropout=0.25,
                         unroll=True),
                         name="encoder")

paragraph_encoding = encoder(paragraph_input_with_wiq)
question_encoding = encoder(question_input_with_wiq)

paragraph_encoding = TimeDistributed(
    Dense(constants.EMBEDDING_DIM,
          use_bias=False,
          trainable=True,
          activation='tanh',
          weights=K.tf.concat([K.tf.eye(constants.EMBEDDING_DIM), K.tf.eye(constants.EMBEDDING_DIM)], axis=1)
    ))(paragraph_encoding)


question_encoding = TimeDistributed(
    Dense(constants.EMBEDDING_DIM,
          use_bias=False,
          trainable=True,
          activation='tanh',
          weights=K.tf.concat([K.tf.eye(constants.EMBEDDING_DIM), K.tf.eye(constants.EMBEDDING_DIM)], axis=1)
    ))(question_encoding)

question_attention_vector = TimeDistributed(Dense(1), name="compute_word_importance_1")(question_encoding)
question_attention_vector = Softmax(axis=1, name="softmax_word_importance1")(question_attention_vector)

question_attention_vector = Multiply(name="apply_on_encoding")([question_encoding, question_attention_vector])
question_attention_vector = Lambda(lambda q: K.sum(q, axis=1), name="compute_single_vector_that_represents_question")(question_attention_vector)
question_attention_vector = RepeatVector(constants.MAX_PARAGRAPH_LENGTH)(question_attention_vector)

answer_start = Concatenate(name="answer_start_variables")([
    paragraph_encoding,
    question_attention_vector,
    Multiply()([paragraph_encoding, question_attention_vector])
])


answer_start = TimeDistributed(Dense(constants.EMBEDDING_DIM, activation='relu'), name="convert_back_to_words_1")(answer_start)
answer_start = TimeDistributed(Dense(1), name="compute_word_importance_2")(answer_start)
answer_start = Flatten()(answer_start)
answer_start = Softmax(name="softmax_word_importance_2")(answer_start)

x = Lambda(lambda x:
   K.tf.cast(K.argmax(x, axis=1), dtype=K.tf.int32),
   name="indices_of_most_answer_starts"
)(answer_start)

start_feature = Lambda(lambda arg:
    K.tf.gather_nd(
        arg[0],
        K.tf.stack([
            K.tf.range(K.tf.shape(arg[1])[0]),
            K.tf.cast(arg[1], K.tf.int32)
        ], axis=1)
    ),
    name="gather_words_from_indices"
)([paragraph_encoding, x])

start_feature = RepeatVector(constants.MAX_PARAGRAPH_LENGTH)(start_feature)

answer_end = Concatenate(name="answer_end_variables")([
    paragraph_encoding,
    question_attention_vector,
    start_feature,
    Multiply()([paragraph_encoding, question_attention_vector]),
    Multiply()([paragraph_encoding, start_feature])
])

answer_end = TimeDistributed(Dense(constants.EMBEDDING_DIM, activation='relu'), name="convert_back_to_words_2")(answer_end)
answer_end = TimeDistributed(Dense(1), name="compute_word_importance_3")(answer_end)
answer_end = Flatten()(answer_end)
answer_end = Softmax(axis=1, name="softmax_word_importance_3")(answer_end)

model = Model(inputs=[paragraph_indices, question_indices], outputs=[answer_start, answer_end])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

def get_model_fit_callbacks():
    callbacks = []

    checkpoint = ModelCheckpoint(
        'weights.{epoch:03d}.hdf5',
        verbose=1,
        save_best_only=False,
        mode='max'
    )

    if constants.IS_LOCAL:
        callbacks = [TensorBoard(
            log_dir='logs',
            histogram_freq=0,
            write_graph=True,
            write_images=False
        ),
        checkpoint]
    else:
        callbacks = [TensorBoard(
            log_dir='gs://squad-bucket/jobs/logs',
            histogram_freq=0,
            write_graph=True,
            write_images=False
        ),
        checkpoint,
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_to_google_cloud(epoch, logs)
        )]

    return callbacks

def save_to_google_cloud(epoch, logs):
    if epoch == 0:
        return

    epochs = '%.03d' % (int(epoch))
    val_loss = '%.02f' % (float(logs['loss']))
    filename = 'weights.' + epochs + '.hdf5'
    filename_with_loss = 'weights.' + epochs + '-' + val_loss + '.hdf5'
    with open(filename, mode='r') as input_f:
        with file_io.FileIO('gs://squad-bucket/models/' + filename_with_loss, mode='w+') as output_f:
            output_f.write(input_f.read())
            print("Saved model.h5 to GCS")
