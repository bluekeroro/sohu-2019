# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
# import lightgbm as lgb
import time
import json
from joblib import load, dump
from tqdm import tqdm
import re
from sprint5_lgb_sklearn.features_ents import feature_ents
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from lightgbm.sklearn import LGBMRegressor
import numpy as np


class Train():
    def __init__(self, train_data_path, model_path, feature_ents_func, debug=False):
        self.train_data_path = train_data_path
        self.model_path = model_path
        self.feature_ents_func = feature_ents_func
        self.debug = debug

    def model_lgb_search(self, X, Y):
        print('model_lgb_search start')
        lgb_model = LGBMRegressor()

        param_grid = dict(
            num_leaves=[50, 60, 70, 80, 90, 100, 120, 130],
            max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13],
            learning_rate=np.linspace(0.03, 0.1, 5),
            n_estimators=[100, 200, 300],
            # num_class=[2],
            # objective=['multi:softmax']  # 'binary:logistic'
        )
        start = time.time()
        cv_split = StratifiedKFold(n_splits=5, shuffle=True)
        grid = GridSearchCV(lgb_model, param_grid, cv=cv_split)  # scoring='neg_log_loss'
        grid_result = grid.fit(X, Y)
        print("Best: %f using params: %s estimator: %s" % (
            grid_result.best_score_, grid_result.best_params_, grid_result.best_estimator_))
        print('GridSearchCV process use %.2f seconds' % (time.time() - start))
        print("Save model to " + self.model_path)
        dump(grid_result, self.model_path)
        print('end=======')

    def model_lgb(self, X, Y):
        # create dataset for lightgbm

        # specify your configurations as a dict
        # params = {
        #     'task': 'train',
        #     'boosting_type': 'gbdt',  # 可换为rf(随机森林) dart goss
        #     'objective': 'binary',
        #     'metric': {'cross_entropy'},  # cross_entropy
        #     'num_leaves': 80,  # 50
        #     # 'max_depth': 6,  # 6
        #     'learning_rate': 0.06,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'seed': 0,
        #     # 'min_data_in_leaf ': 100,
        # }  # f1 0.43
        # train
        lgb_model = LGBMRegressor()
        print("Training lgb model....")
        gbm = lgb_model.fit(X, Y)
        print("feature_importances_ : ", gbm.feature_importances_)
        print("Save model to " + self.model_path)
        dump(gbm, self.model_path)

    def train_ents(self):
        # train_data = open(self.train_data_path, 'r', encoding='utf-8').readlines()
        train_data = list()
        with open(self.train_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                train_data.append(line)

        if self.debug is True:
            train_data = train_data[:int(len(train_data) / 10 / 10)]
        X = []
        Y = []
        cnt = 0
        cntSum = 0
        for news in tqdm(train_data):
            news = json.loads(news)
            X_data = self.feature_ents_func.combine_features(news)
            Y_data = [x['entity'] for x in news['coreEntityEmotions']]
            cntSum += len(Y_data)
            for x in X_data:
                if x[0][0] in Y_data:
                    cnt += 1
                    Y.append(1)
                else:
                    Y.append(0)
                X.append(x[1])
        print("结巴分词准确率：{}%".format(cnt / cntSum * 100))
        print("Save features... ")
        dump(X, "models/x1_featrues.joblib")
        dump(Y, "models/y1_featrues.joblib")
        # X = load("models/x1_featrues.joblib")
        # Y = load("models/y1_featrues.joblib")
        # self.model_lgb(X, Y)
        self.model_lgb_search(X, Y)
        print("done!")


if __name__ == "__main__":
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model1.joblib', feature_ents_func)
    train.train_ents()
