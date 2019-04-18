# -*- coding:UTF-8 -*-
"""
@File    : f1_score.py
@Time    : 2019/4/16 16:08
@Author  : Blue Keroro
"""
import sklearn as sk
from sklearn.metrics import confusion_matrix
import codecs
import json
import numpy as np
import tqdm


def computeF1Score(true_path, pred_path):
    predData = loadPredData(pred_path)
    # predData = loadTrueData(pred_path)
    trueData = loadTrueData(true_path)
    temp = {}
    for newsId in predData:
        temp[newsId] = trueData[newsId]
    trueData = temp
    trueDataArray = np.zeros((len(trueData), 3))
    predDataEntityArray = np.zeros((len(trueData), 3))
    predDataEmotionArray = np.zeros((len(trueData), 3))
    for row, newsid in enumerate(trueData):
        # print(row/len(trueData)*100,'%')
        if 'entity' in trueData[newsid]:
            size = len(trueData[newsid]['entity'])
            trueDataArray[row][:size] = 1

        if 'entity' not in trueData[newsid] or 'entity' not in predData[newsid]:
            predDataEntityArray[row] = 0
            predDataEmotionArray[row] = 0
        else:
            for col, entity in enumerate(predData[newsid]['entity']):
                if col>2:
                    break
                if entity in trueData[newsid]['entity']:
                    predDataEntityArray[row][col] = 1
                    try:
                        if trueData[newsid]['emotion'][col] == predData[newsid]['emotion'][col]:
                            predDataEmotionArray[row][col] = 1
                    except IndexError:
                        predDataEmotionArray[row][col] = 0

    trueDataArray = np.reshape(trueDataArray, [-1])
    predDataEntityArray = np.reshape(predDataEntityArray, [-1])
    predDataEmotionArray = np.reshape(predDataEmotionArray, [-1])
    entityScore = sk.metrics.f1_score(trueDataArray, predDataEntityArray, average='binary')
    emotionScore = sk.metrics.f1_score(trueDataArray, predDataEmotionArray, average='binary')
    return entityScore, emotionScore


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
    return data


def loadPredData(filePath):
    data = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip().split("\t")
                data[line[0]] = {}
                data[line[0]]['entity'] = line[1].split(',')
                data[line[0]]['emotion'] = line[2].split(',')
            except IndexError:
                pass
    return data


if __name__ == '__main__':
    entityScore, emotionScore = computeF1Score('coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                               'coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
    print('entityScore:', entityScore)
    print('emotionScore:', emotionScore)
