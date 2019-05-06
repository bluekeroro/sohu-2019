# -*- coding:UTF-8 -*-
"""
@File    : test_entity_number.py
@Time    : 2019/5/3 21:50
@Author  : Blue Keroro
"""
from f1_score import *
import random

if __name__ == '__main__':
    true_data = loadTrueData('data/coreEntityEmotion_train.txt')
    pred_data = loadPredData('data/2_coreEntityEmotion_train_result.txt')
    cntNumPred = [0, 0, 0]
    cntNumTrue = [0, 0, 0]
    for newsId in pred_data:
        cntNumPred[len(pred_data[newsId]['entity']) - 1] += 1
        cntNumTrue[len(true_data[newsId]['entity']) - 1 if len(true_data[newsId]['entity']) < 4 else 2] += 1
    print('cntNumPred:', cntNumPred)
    print('cntNumTrue:', cntNumTrue)
    print('误差率：{}%'.format([abs(cntNumPred[i] - cntNumTrue[i]) / cntNumTrue[i]*100 for i in range(3)]))
