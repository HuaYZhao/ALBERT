import json
from copy import deepcopy
import os
from matplotlib import pyplot as plt
from data.eval import get_raw_scores
import numpy as np
import time

prediction_file = './data/predictions.json'
prediction = json.load(open(prediction_file, 'r', encoding='utf-8'))

null_odds_file = './data/null_odds.json'
null_odds = json.load(open(null_odds_file, 'r', encoding='utf-8'))

nbest_predictions_file = './data/nbest_predictions.json'
nbest_predictions = json.load(open(nbest_predictions_file, 'r', encoding='utf-8'))

no_answer_prediction_file = './data/no_answer_predictions.json'
no_answer_prediction = json.load(open(no_answer_prediction_file, 'r', encoding='utf-8'))

dev_file = './data/dev-v2.0.json'
dev = json.load(open(dev_file, 'r', encoding='utf-8'))['data']

merge_prediction_file = './data/merge_predictions.json'


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


qid_to_has_ans = make_qid_to_has_ans(json.load(open('./data/dev-v2.0.json', 'r', encoding='utf-8'))["data"])


def simple_replace():
    y = list(no_answer_prediction.values())

    exact_scores, f1_scores = get_raw_scores(dev, prediction)
    best_exact = sum(exact_scores.values()) / len(exact_scores)
    best_f1 = sum(f1_scores.values()) / len(f1_scores)
    best_threshold = None
    best_merge_prediction = None

    def threshold_merge(threshold=1.):
        merge_prediction = deepcopy(prediction)
        for q_ids in prediction.keys():
            if no_answer_prediction[q_ids] > threshold:
                merge_prediction[q_ids] = ''
        return merge_prediction

    x = np.arange(0., 2., 0.01)
    y1 = list()
    y2 = list()
    y3 = list()
    for threshold in x:
        merge_prediction = threshold_merge(threshold)
        exact_scores, f1_scores = get_raw_scores(dev, merge_prediction)
        exact = sum(exact_scores.values()) / len(exact_scores)
        f1 = sum(f1_scores.values()) / len(f1_scores)
        y1.append(exact)
        y2.append(f1)
        y3.append((exact + f1) / 2)
        print(threshold, exact, f1, (exact + f1) / 2)
        if exact + f1 > best_exact + best_f1:
            best_exact = exact
            best_f1 = f1
            best_threshold = threshold
            best_merge_prediction = deepcopy(merge_prediction)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()

    json.dump(best_merge_prediction, open(merge_prediction_file, 'w', encoding='utf-8'), indent=4)
    print(f"best_threshold: {best_threshold}")

    # plt.boxplot(y)
    # plt.show()


def simple_replace_with_null_odds():
    y = list(no_answer_prediction.values())

    exact_scores, f1_scores = get_raw_scores(dev, prediction)
    best_exact = sum(exact_scores.values()) / len(exact_scores)
    best_f1 = sum(f1_scores.values()) / len(f1_scores)
    best_threshold = None
    best_alpha = None
    best_merge_prediction = None

    def threshold_merge(alpha=0.5, threshold=1.):
        merge_prediction = deepcopy(prediction)
        for q_ids in prediction.keys():
            null = alpha * no_answer_prediction[q_ids] + (1 - alpha) * null_odds[q_ids]
            if null > threshold:
                merge_prediction[q_ids] = ''
        return merge_prediction

    for alpha in np.arange(0., 1., 0.1):
        for threshold in np.arange(-2., 3., 0.1):
            merge_prediction = threshold_merge(alpha, threshold)
            exact_scores, f1_scores = get_raw_scores(dev, merge_prediction)
            exact = sum(exact_scores.values()) / len(exact_scores)
            f1 = sum(f1_scores.values()) / len(f1_scores)
            print(alpha, threshold, exact, f1, (exact + f1) / 2)
            if exact + f1 > best_exact + best_f1:
                best_exact = exact
                best_f1 = f1
                best_alpha = alpha
                best_threshold = threshold
                best_merge_prediction = deepcopy(merge_prediction)

    for alpha in np.arange(best_alpha - 0.1, best_alpha + 0.1, 0.01):
        for threshold in np.arange(best_threshold - 0.3, best_threshold + 0.3, 0.01):
            merge_prediction = threshold_merge(alpha, threshold)
            exact_scores, f1_scores = get_raw_scores(dev, merge_prediction)
            exact = sum(exact_scores.values()) / len(exact_scores)
            f1 = sum(f1_scores.values()) / len(f1_scores)
            print(alpha, threshold, exact, f1, (exact + f1) / 2)
            if exact + f1 > best_exact + best_f1:
                best_exact = exact
                best_f1 = f1
                best_alpha = alpha
                best_threshold = threshold
                best_merge_prediction = deepcopy(merge_prediction)

    json.dump(best_merge_prediction, open(merge_prediction_file, 'w', encoding='utf-8'), indent=4)
    print(f"best_alpha: {best_alpha}"
          f"best_threshold: {best_threshold}")


def prob_statistic_with_nbest():
    y = list(no_answer_prediction.values())
    exact_scores, f1_scores = get_raw_scores(dev, prediction)
    best_exact = sum(exact_scores.values()) / len(exact_scores)
    best_f1 = sum(f1_scores.values()) / len(f1_scores)
    best_threshold1 = None
    best_threshold2 = None
    best_merge_prediction = None

    def merge_with_nbest_prob(threshold1=0.5, threshold2=0.5):
        merge_prediction = deepcopy(prediction)
        for q_ids, answers in nbest_predictions.items():
            nbest_answer = sorted(answers, key=lambda o: o["probability"])[-1]
            if no_answer_prediction[q_ids] > threshold1 and nbest_answer["probability"] < threshold2:
                merge_prediction[q_ids] = ''
        return merge_prediction

    for threshold1 in np.arange(1.26, 1.27, 0.01):
        for threshold2 in np.arange(-1., 2., 0.1):
            merge_prediction = merge_with_nbest_prob(threshold1, threshold2)
            exact_scores, f1_scores = get_raw_scores(dev, merge_prediction)
            exact = sum(exact_scores.values()) / len(exact_scores)
            f1 = sum(f1_scores.values()) / len(f1_scores)
            print(threshold1, threshold2, exact, f1, (exact + f1) / 2)

            if exact + f1 > best_exact + best_f1:
                best_exact = exact
                best_f1 = f1
                best_threshold1 = threshold1
                best_threshold2 = threshold2
                best_merge_prediction = deepcopy(merge_prediction)

    json.dump(best_merge_prediction, open(merge_prediction_file, 'w', encoding='utf-8'), indent=4)
    print(f"best_threshold: {best_threshold1} {best_threshold2}")


xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/predictions.json"
os.system(xargs)

start = time.time()
simple_replace_with_null_odds()
xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/merge_predictions.json"
os.system(xargs)
print(f"cost time: {time.time() - start}")
