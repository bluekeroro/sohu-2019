# -*- coding:UTF-8 -*-
"""
@File    : main.py
@Time    : 2019/4/17 16:05
@Author  : Blue Keroro
"""
from sprint2.test import Test
from data_split import data_split
from f1_score import computeF1Score
from sprint2.train import Train

if __name__ == '__main__':
    from time import time

    start = time()
    test = Test()
    turn = 5
    entityScoreSum = 0
    emotionScoreSum = 0
    for i in range(turn):
        # 切分数据
        data_split('../coreEntityEmotion_baseline/data', 'coreEntityEmotion_train.txt')
        # 训练
        # trainer = Train()
        # trainer.trainCoreEntity()
        # trainer.trainEmotion()
        # 测试
        test.testCoreEntity('../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train.txt',
                            '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
        # 计算F1
        entityScore, emotionScore = computeF1Score('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                                                   '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
        print('turn:', i + 1, 'entityScore:', entityScore, 'emotionScore:', emotionScore)
        # 统计F1
        entityScoreSum += entityScore
        emotionScoreSum += emotionScore
    # 输出平均值
    print('平均entityScore:', entityScoreSum / turn, '平均emotionScore:', emotionScoreSum / turn)

    print('end:', time() - start)
