import json
from copy import deepcopy
import os
import numpy as np

prediction_file = './data/predictions.json'
prediction = json.load(open(prediction_file, 'r', encoding='utf-8'))

null_odds_file = './data/null_odds.json'
null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))

nbest_predictions_file = './data/nbest_predictions.json'
nbest_predictions = json.load(open(nbest_predictions_file, 'r', encoding='utf-8'))

no_answer_prediction_file = './data/no_answer_predictions.json'
no_answer_prediction = json.load(open(no_answer_prediction_file, 'r', encoding='utf-8'))

merge_prediction_file = './data/merge_predictions.json'


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


qid_to_has_ans = make_qid_to_has_ans(json.load(open('./data/dev-v2.0.json', 'r', encoding='utf-8'))["data"])
merge_prediction = deepcopy(prediction)


def simple_replace():
    for q_ids in prediction.keys():
        if qid_to_has_ans[q_ids] and no_answer_prediction[q_ids] > 0.5:
            print(no_answer_prediction[q_ids])

        prob = 1 / (1 + np.exp(-no_answer_prediction[q_ids]))

        if prob > 0.5:
            merge_prediction[q_ids] = ''

    json.dump(merge_prediction, open(merge_prediction_file, 'w', encoding='utf-8'), indent=4)


def find_threshold():
    pass


xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/predictions.json"
os.system(xargs)

xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/merge_predictions.json"
os.system(xargs)
