# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
import xgboost as xgb
import lightgbm as lgb
import time
import json
from joblib import load, dump
from tqdm import tqdm
import numpy as np
import re
from sprint_multi_model_test.features_ents import feature_ents
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, ShuffleSplit, \
    StratifiedKFold
from xgboost.sklearn import XGBRegressor


class Train():
    def __init__(self, train_data_path, model_path_lgb,model_path_xgb, feature_ents_func, debug=False):
        self.train_data_path = train_data_path
        self.model_path_lgb = model_path_lgb
        self.model_path_xgb = model_path_xgb
        self.feature_ents_func = feature_ents_func
        self.debug = debug

    def model_xgb(self, X, Y):
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.1, random_state=0)  # 分训练集和验证集
        xgb_train = xgb.DMatrix(train_x, label=train_y)
        xgb_eval = xgb.DMatrix(valid_x, label=valid_y)

        # specify your configurations as a dict
        # params = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}  # 0.421
        params = {'booster': 'gbtree','eta': 0.138, 'max_depth': 2, 'n_estimators': 100,'silent': 0,
                  'objective': 'binary:logistic'} # # 0.439
        watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]
        # train
        print("Training xgb model....")
        evals_result = {}
        gbm = xgb.train(params, xgb_train, evals=watchlist, evals_result=evals_result, num_boost_round=100,
                        early_stopping_rounds=10)
        print('xgb 训练结果 evals_result：', evals_result)
        print("Save model to " + self.model_path_xgb)
        dump(gbm, self.model_path_xgb)

    def model_lgb(self, X, Y):
        # create dataset for lightgbm
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.1, random_state=0)  # 分训练集和验证集
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 可换为rf(随机森林) dart goss
            'objective': 'binary',
            'metric': {'cross_entropy'},  # cross_entropy
            'num_leaves': 50,
            'max_depth': 6,  # 3
            'learning_rate': 0.06,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 0,
            # 'min_data_in_leaf ': 100,
        }  # f1 0.43
        # train
        evals_result_dict = {}
        print("Training lgb model....")
        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_eval, lgb_train], early_stopping_rounds=10,
                        valid_names=['eval', 'train'], evals_result=evals_result_dict)
        print('lgb 训练结果 evals_result：', evals_result_dict)
        print("Save model to " + self.model_path_lgb)
        dump(gbm, self.model_path_lgb)

    def train_ents(self,load_model=False):
        X = []
        Y = []
        if load_model is False:
            train_data = open(self.train_data_path, 'r', encoding='utf-8').readlines()
            if self.debug is True:
                train_data = train_data[:int(len(train_data) / 10 / 10)]
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
        else:
            X = load("models/x1_featrues.joblib")
            Y = load("models/y1_featrues.joblib")
        # X = load("models/x1_featrues.joblib")
        # Y = load("models/y1_featrues.joblib")
        self.model_lgb(X, Y)
        self.model_xgb(X, Y)
        # self.model_xgb_search(X, Y)
        print("done!")


if __name__ == "__main__":
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model_test_lgb.joblib', 'models/model1_test_xgb.joblib', feature_ents_func)
    train.train_ents()
