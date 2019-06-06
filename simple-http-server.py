#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import json
from io import open
from training import preprocessing
from keras.models import Model, load_model
from keras.layers import merge, concatenate, multiply
from keras import backend as K
from training import constants
from training import fast_qa
from training import glove_embeddings
from training import preprocessing
import numpy as np
from io import open
from keras.utils import to_categorical

model = load_model('models/model-4.hdf5', custom_objects={
    'backend': K,
    'concatenate': concatenate,
    'multiply': multiply,
    'wiq_b': fast_qa.wiq_b,
    'wiq_w': fast_qa.wiq_w,
    'get_question_wiq': fast_qa.get_question_wiq
})

inverted = {v: k for k, v in glove_embeddings.word2idx.iteritems()}

print('Model loaded.')

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(open('index.html', 'r').read())

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = json.loads(self.rfile.read(content_length))

        e_paragraph, e_question = preprocessing.get_encoded_input(post_data["paragraph"], post_data["question"])
        paragraph = np.array(e_paragraph)
        question = np.array(e_question)

        paragraph.resize(constants.MAX_PARAGRAPH_LENGTH, refcheck=False)
        question.resize(constants.MAX_QUESTION_LENGTH, refcheck=False)

        paragraphs = np.concatenate([[paragraph], np.zeros((39, constants.MAX_PARAGRAPH_LENGTH))], axis=0)
        questions = np.concatenate([[question], np.zeros((39, constants.MAX_QUESTION_LENGTH))], axis=0)

        predictions = model.predict([paragraphs, questions], verbose=1, batch_size=40)

        start = np.argmax(predictions[0][0], axis=0)
        end = np.argmax(predictions[1][0], axis=0)

        answer = e_paragraph[start:end+1]
        result = [inverted[index] for index in answer]

        self._set_headers()
        self.wfile.write(" ".join(result))

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
