# -*- coding:UTF-8 -*-
"""
@File    : data_analyzed.py
@Time    : 2019/4/25 17:51
@Author  : Blue Keroro
"""
from f1_score import *


def analyz():
    pass


if __name__ == '__main__':
    entityScore, emotionScore = computeF1Score('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                               'result/result_lgb.txt')
    print('lgb：', 'entityScore:', entityScore, 'emotionScore:', emotionScore)
    entityScore, emotionScore = computeF1Score('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                               'result/result_xgb.txt')
    print('xgb：', 'entityScore:', entityScore, 'emotionScore:', emotionScore)
    trueData = loadTrueData('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt')
    lgb_pred_data = loadPredData('result/result_lgb.txt')
    xgb_pred_data = loadPredData('result/result_xgb.txt')
    lgb_ent_set = set()
    xgb_ent_set = set()
    cnt = 0
    for newsid in lgb_pred_data:
        cnt += 1
        for entity in lgb_pred_data[newsid]['entity']:
            if entity in trueData[newsid]['entity']:
                lgb_ent_set.add(newsid + ' ' + entity)
        for entity in xgb_pred_data[newsid]['entity']:
            if entity in trueData[newsid]['entity']:
                xgb_ent_set.add(newsid + ' ' + entity)
    print("总的实体量：", cnt * 3)
    print("lgb预测正确的实体量：",len(lgb_ent_set)/cnt / 3*100,'%')
    print("xgb预测正确的实体量：", len(xgb_ent_set)/cnt / 3*100,'%')
    print("共同预测正确的实体量：", len(lgb_ent_set&xgb_ent_set)/cnt / 3*100,'%')
