# -*- coding:UTF-8 -*-
"""
@File    : main.py
@Time    : 2019/4/17 16:05
@Author  : Blue Keroro
"""
from sprint5.features_ents import feature_ents
from sprint5.test import Test
from data_split import data_split
from f1_score import computeF1Score
from sprint5.train import Train


def accuracy(train, test):
    turn = 4
    entityScoreSum = 0
    emotionScoreSum = 0
    for i in range(turn):
        # 切分数据
        data_split('../coreEntityEmotion_baseline/data', 'coreEntityEmotion_train.txt')
        # 训练
        train.train_ents()
        # 测试
        test.test()
        # 计算F1
        entityScore, emotionScore = computeF1Score('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                                   '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
        print('turn:', i + 1, 'entityScore:', entityScore, 'emotionScore:', emotionScore)
        # 统计F1
        entityScoreSum += entityScore
        emotionScoreSum += emotionScore
    # 输出平均值
    print('平均entityScore:', entityScoreSum / turn, '平均emotionScore:', emotionScoreSum / turn)
    return entityScoreSum / turn, emotionScoreSum / turn


if __name__ == '__main__':
    from time import time

    start = time()
    model_path = 'models/model1.joblib'
    feature_ents_func= feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                 '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/8_coreEntityEmotion_train.txt',
                  model_path,feature_ents_func,debug=True)
    test = Test('../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train.txt',
                '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt',
                model_path,feature_ents_func,debug=True)
    accuracy(train,test)
    print('end:', time() - start)
