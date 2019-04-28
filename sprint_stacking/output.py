# -*- coding:UTF-8 -*-
"""
@File    : output.py
@Time    : 2019/4/18 14:43
@Author  : Blue Keroro
"""
from time import time

from sprint_stacking.features_ents import feature_ents
from sprint_stacking.test import Test
from sprint_stacking.train import Train

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