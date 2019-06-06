import glove_embeddings
import json
import evaluate
import constants
import pprint
import re
import numpy as np
from io import open
from tensorflow.python.lib.io import file_io

# https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
def find_sublist(list, sublist):
    results = []
    sublist_length = len(sublist)
    for index in (i for i, e in enumerate(list) if e == sublist[0]):
        if list[index:index + sublist_length] == sublist:
            results.append((index, index + sublist_length - 1))
    return results

def load_dataset(name):
    if constants.IS_LOCAL:
        file = open(name, "r", encoding='utf-8').read()
    else:
        file = file_io.FileIO('gs://squad-bucket/' + name, mode='r').read()

    dataset = json.loads(file)
    data = dataset['data']

    datapoints = []

    for topic in data:
        datapoints += [{
            'paragraph':    paragraph['context'],
            'id':            qa['id'],
            'question':      qa['question'],
            'answer':        qa['answers'][0]['text'],
            'answer_start':  qa['answers'][0]['answer_start'],
            'topic':         topic['title'] }
            for paragraph in topic['paragraphs']
            for qa in paragraph['qas']]

    return datapoints

def get_updated_answer_indices(entry):
    all_occurrences = [match.start() for match in re.finditer(re.escape(entry['answer']), entry['paragraph'])]
    answer_start = all_occurrences.index(entry['answer_start'])

    normal_p = evaluate.normalize_answer(entry['paragraph']).split()
    normal_a = evaluate.normalize_answer(entry['answer']).split()

    return find_sublist(normal_p, normal_a)[answer_start]

def get_encoded_input(paragraph, question):
    paragraph = evaluate.normalize_answer(paragraph).split()
    question = evaluate.normalize_answer(question).split()

    if len(paragraph) > constants.MAX_PARAGRAPH_LENGTH:
        return

    paragraph = [glove_embeddings.get_word_index(word) for word in paragraph]
    question = [glove_embeddings.get_word_index(word) for word in question]

    return (paragraph, question)

def preprocess(data):
    processed_data = []
    for text in data:
        try:
            answer_span = get_updated_answer_indices(text)

            paragraph = evaluate.normalize_answer(text['paragraph']).split()
            question = evaluate.normalize_answer(text['question']).split()

            if len(paragraph) > constants.MAX_PARAGRAPH_LENGTH:
                continue

            paragraph = [glove_embeddings.get_word_index(word) for word in paragraph]
            question = [glove_embeddings.get_word_index(word) for word in question]
        except IndexError:
            continue

        processed_data += [{
            'paragraph': paragraph,
            'question': question,
            'answer_start': answer_span[0],
            'answer_end': answer_span[1]
        }]

    return processed_data

def load_preprocessed_dataset(name):
    if constants.IS_LOCAL:
        file = open(name, "r", encoding='utf-8').read()
    else:
        file = file_io.FileIO('gs://squad-bucket/' + name, mode='r').read()

    dataset = json.loads(file)

    paragraphs = []
    questions = []
    answer_starts = []
    answer_ends = []

    for datapoint in dataset:
        paragraph = np.array(datapoint['paragraph'])
        question = np.array(datapoint['question'])
        paragraph.resize(constants.MAX_PARAGRAPH_LENGTH, refcheck=False)
        question.resize(constants.MAX_QUESTION_LENGTH, refcheck=False)
        paragraphs.append(paragraph)
        questions.append(question)
        answer_starts.append(datapoint['answer_start'])
        answer_ends.append(datapoint['answer_end'])

    return (paragraphs, questions, answer_starts, answer_ends)

def save(name, data):
    if constants.IS_LOCAL:
        with open(name, mode='w+', encoding='utf-8') as output_f:
            output_f.write(unicode(json.dumps(data)))
            print("Saved " + name + " to local FS.")
    else:
        with file_io.FileIO('gs://squad-bucket/' + name, mode='w+') as output_f:
            output_f.write(unicode(json.dumps(data)))
            print("Saved " + name + " to GCS")
