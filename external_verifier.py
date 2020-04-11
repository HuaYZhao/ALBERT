import json
from copy import deepcopy
import os
from matplotlib import pyplot as plt
from data.eval import get_raw_scores, normalize_answer
import numpy as np
import time
from functional import seq

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

train_file = './data/train-v2.0.json'
train = json.load(open(train_file, 'r', encoding='utf-8'))['data']

train_preds_file = './data/squad_preds.json'
train_preds = json.load(open(train_preds_file, 'r', encoding='utf-8'))

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
        for threshold in np.arange(-4., 1., 0.1):
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


def check_data():
    dev_exact_scores, dev_f1_scores = get_raw_scores(dev, prediction)
    train_exact_scores, train_f1_scores = get_raw_scores(train, train_preds)

    train_error_exact = seq(train_exact_scores.items()).filter(lambda x: x[1] == 0).map(lambda x: x[0]).list()
    dev_error_exact = seq(dev_exact_scores.items()).filter(lambda x: x[1] == 0).map(lambda x: x[0]).list()

    all_dev_qas = seq(dev).map(lambda x: x['paragraphs']).flatten().list()
    all_train_qas = seq(train).map(lambda x: x['paragraphs']).flatten().list()

    def find_id(id_, all_qas):
        for qas in all_qas:
            qa = seq(qas['qas']).filter(lambda x: x['id'] == id_).list()
            if qa:
                return dict(context=qas['context'],
                            prediction=prediction.get(id_, None) or train_preds.get(id_, None),
                            **qa[0])

    dev_error_exact_qas = [find_id(id_, all_dev_qas) for id_ in dev_error_exact[:100]]
    train_error_exact_qas = [find_id(id_, all_train_qas) for id_ in train_error_exact[:100]]

    # x = seq(dev).map(lambda x: [x['title'], get_raw_scores([x], prediction)]).map(
    #     lambda x: [x[0], np.mean(list(x[1][0].values())), np.mean(list(x[1][1].values()))]).list()
    print(1)


def write_answer_refine():
    answer_refine_file = './data/answer_refine.meta'

    answer_refine_class = {}
    for article in train:
        for p in article['paragraphs']:
            for qa in p['qas']:
                info = {}
                qid = qa['id']
                info['pred'] = train_preds[qid]

                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                assert len(gold_answers) <= 1, gold_answers
                if not gold_answers:
                    info['answer'] = ""
                    info['class'] = 0
                    answer_refine_class[qid] = info
                    continue
                gold_answer = gold_answers[0]
                info['answer'] = gold_answer
                pred = train_preds[qid]
                if pred == gold_answer:
                    info['class'] = 0
                elif pred in gold_answer:
                    info['class'] = 1
                elif gold_answer in pred:
                    info['class'] = 2
                else:
                    info['class'] = 0
                answer_refine_class[qid] = info
    json.dump(answer_refine_class, open(answer_refine_file, 'w', encoding='utf-8'))


# xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/predictions.json"
# os.system(xargs)

# start = time.time()
# simple_replace_with_null_odds()
# xargs = "python ./data/eval.py ./data/dev-v2.0.json ./data/merge_predictions.json"
# os.system(xargs)
# print(f"cost time: {time.time() - start}")
write_answer_refine()
