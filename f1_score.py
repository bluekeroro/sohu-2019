# -*- coding:UTF-8 -*-
"""
@File    : f1_score.py
@Time    : 2019/4/16 16:08
@Author  : Blue Keroro
"""
import codecs
import json


def computeF1Score(true_path, pred_path):
    predData = loadPredData(pred_path)
    # predData = loadTrueData(pred_path)
    trueData = loadTrueData(true_path)
    entityScoreSum = 0
    emotionScoreSum = 0
    for newsid in predData:
        # print(row/len(trueData)*100,'%')
        entityScore = f1_score(trueData[newsid]['entity'], predData[newsid]['entity'])
        emotionScore = f1_score(trueData[newsid]['emotion'], predData[newsid]['emotion'])
        entityScoreSum += entityScore
        emotionScoreSum += emotionScore
    return entityScoreSum / len(predData), emotionScoreSum / len(predData)


def loadTrueData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = {}
    for line in f:
        news = json.loads(line.strip())
        data[news['newsId']] = {}
        data[news['newsId']]['entity'] = []
        data[news['newsId']]['emotion'] = []
        for coreEntityEmotion in news['coreEntityEmotions']:
            data[news['newsId']]['entity'].append(coreEntityEmotion['entity'])
            data[news['newsId']]['emotion'].append(coreEntityEmotion['emotion'])
        if len(data[news['newsId']]['entity']) != len(data[news['newsId']]['emotion']):
            raise Exception("实体与情感数量不等", "newsId:", news['newsId'])
        tmp = []
        for index in range(len(data[news['newsId']]['entity'])):
            tmp.append(data[news['newsId']]['entity'][index] + '_' + data[news['newsId']]['emotion'][index])
        data[news['newsId']]['emotion'] = tmp
    return data


def loadPredData(filePath):
    data = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            data[line[0]] = {}
            data[line[0]]['entity'] = line[1].split(',')
            data[line[0]]['emotion'] = line[2].split(',')
            if len(data[line[0]]['entity']) != len(data[line[0]]['emotion']):
                raise Exception("实体与情感数量不等", "newsId:", line[0])
            tmp = []
            for index in range(len(data[line[0]]['entity'])):
                tmp.append(data[line[0]]['entity'][index] + '_' + data[line[0]]['emotion'][index])
            data[line[0]]['emotion'] = tmp
    return data


def precision_score(y_true, y_pred):
    cnt = 0
    for i in y_pred:
        if i in y_true:
            cnt += 1
    return cnt / len(y_pred)


def recall_score(y_true, y_pred):
    cnt = 0
    for i in y_pred:
        if i in y_true:
            cnt += 1
    return cnt / len(y_true)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    # y_true = ['a', 'b', 'c']
    # y_pred = ['a', 'b', 'c']
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    #
    # print(precision, recall, f1_score(y_true, y_pred))
    # y_true = ['d', 'e', 'f']
    # y_pred = ['d', 'e']
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # print(precision, recall, f1_score(y_true, y_pred))
    #
    # y_true = ['a_pos', 'b_pos', 'c_pos']
    # y_pred = ['a_pos', 'b_pos', 'c_neg']
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # print(precision, recall, f1_score(y_true, y_pred))
    #
    # y_true = ['d_neg', 'e_neg', 'f_neg']
    # y_pred = ['d_neg', 'e_pos']
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # print(precision, recall, f1_score(y_true, y_pred))

    entityScore, emotionScore = computeF1Score('coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                               'coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
    print('entityScore:', entityScore)
    print('emotionScore:', emotionScore)
