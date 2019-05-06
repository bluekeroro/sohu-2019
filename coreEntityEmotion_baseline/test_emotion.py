# -*- coding:UTF-8 -*-
"""
@File    : test_emotion.py
@Time    : 2019/5/3 1:54
@Author  : Blue Keroro
"""
from f1_score import *
import random

if __name__ == '__main__':
    true_data = loadTrueData('data/coreEntityEmotion_train.txt')
    pred_data = loadPredData('data/2_coreEntityEmotion_train_result.txt')

    entityScore, emotionScore = computeF1Score('data/coreEntityEmotion_train.txt',
                                               'data/2_coreEntityEmotion_train_result.txt')
    print("before:")
    print('entityScore:', entityScore)
    print('emotionScore:', emotionScore)
    # *消极实体百分比12.628095766971617 %
    # *中立实体百分比37.94153680570663 %
    # *积极实体百分比49.430367427321755 %
    probilityNEG = 0.1262
    probilityNORM = 0.3794
    probilityPOS = 0.4943
    with open('data/2_coreEntityEmotion_train_emotion_result.txt','w',encoding='utf-8') as file:
        for newsId in pred_data:
            entity_list = []
            emotion_list = []
            for emotion in pred_data[newsId]['emotion']:
                random_probility = random.random()
                word_emotion = emotion.split('_')
                if len(word_emotion) >2:
                    raise Exception("情感中含有多个下划线  newsId：",newsId)
                entity_list.append(word_emotion[0])
                if random_probility <= probilityNEG:
                    word_emotion[1] = 'NEG'
                elif random_probility > 1-probilityPOS:
                    word_emotion[1] = 'POS'
                else:
                    word_emotion[1] = 'NORM'
                emotion_list.append(word_emotion[1])
            file.write("{}\t{}\t{}\n".format(newsId,','.join(entity_list),','.join(emotion_list)))
    entityScore, emotionScore = computeF1Score('data/coreEntityEmotion_train.txt',
                                               'data/2_coreEntityEmotion_train_emotion_result.txt')
    print("after:")
    print('entityScore:', entityScore)
    print('emotionScore:', emotionScore)
