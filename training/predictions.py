from keras.models import Model, load_model
from keras.layers import merge, concatenate, multiply
from keras import backend as K
import constants
import fast_qa
import preprocessing
import numpy as np
from io import open
import json
from keras.utils import to_categorical

def evaluate():
    model = load_model('models/model-4.hdf5', custom_objects={
        'backend': K,
        'concatenate': concatenate,
        'multiply': multiply,
        'wiq_b': fast_qa.wiq_b,
        'wiq_w': fast_qa.wiq_w,
        'get_question_wiq': fast_qa.get_question_wiq
    })

    print('Model loaded.')
    paragraphs, questions, answer_starts, answer_ends = preprocessing.load_preprocessed_dataset('processed-dev-v1.1.json')

    #answer_starts = to_categorical(answer_starts, constants.MAX_PARAGRAPH_LENGTH)
    #answer_ends = to_categorical(answer_ends, constants.MAX_PARAGRAPH_LENGTH)

    print('Evaluating...')
    predictions = model.predict([paragraphs[:10000], questions[:10000]], verbose=1, batch_size=40)

    starts = np.argmax(predictions[0], axis=1)
    ends = np.argmax(predictions[1], axis=1)

    with open("starts.json", mode='w+', encoding='utf-8') as output_f:
        output_f.write(unicode(json.dumps(starts.tolist())))

    with open("ends.json", mode='w+', encoding='utf-8') as output_f:
        output_f.write(unicode(json.dumps(ends.tolist())))

    print('Written.')
evaluate()
