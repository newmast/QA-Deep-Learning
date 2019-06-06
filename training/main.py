import numpy as np
import constants
import glove_embeddings
import preprocessing
from keras.utils import to_categorical

paragraphs, questions, answer_starts, answer_ends = preprocessing.load_preprocessed_dataset('processed-train-v1.1.json')

answer_starts = to_categorical(answer_starts, constants.MAX_PARAGRAPH_LENGTH)
answer_ends = to_categorical(answer_ends, constants.MAX_PARAGRAPH_LENGTH)

fast_qa.model.fit(
    x=[paragraphs, questions],
    y=[answer_starts, answer_ends],
    batch_size=512,
    epochs=100,
    shuffle=True,
    callbacks=fast_qa.get_model_fit_callbacks())  # starts training
