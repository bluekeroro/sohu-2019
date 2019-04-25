# -*- coding:UTF-8 -*-
"""
@File    : output.py
@Time    : 2019/4/18 14:43
@Author  : Blue Keroro
"""
from time import time

from sprint6.features_ents import feature_ents
from sprint6.test import Test
from sprint6.train import Train

if __name__ == '__main__':
    start = time()
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model_xgb.joblib', feature_ents_func)
    train.train_ents()
    test = Test('../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1.txt',
                '../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1_result.txt',
                'models/model_xgb.joblib', feature_ents_func)
    test.test()
    print('end:', time() - start)