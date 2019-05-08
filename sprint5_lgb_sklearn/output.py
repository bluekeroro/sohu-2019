# -*- coding:UTF-8 -*-
"""
@File    : output.py
@Time    : 2019/4/18 14:43
@Author  : Blue Keroro
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from time import time

from sprint5_lgb_sklearn.features_ents import feature_ents
from sprint5_lgb_sklearn.test import Test
from sprint5_lgb_sklearn.train import Train

if __name__ == '__main__':
    start = time()
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt',load_from_file=True)
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model_lgb.joblib', feature_ents_func)
    train.train_ents()
    test = Test('../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1.txt',
                '../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1_result.txt',
                'models/model_lgb.joblib', feature_ents_func)
    test.test()
    print('end:', time() - start)