# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
import lightgbm as lgb
import time
import json
from joblib import load, dump
from tqdm import tqdm
import re
from sprint5.features_ents import feature_ents
from sklearn.model_selection import train_test_split


class Train():
    def __init__(self,train_data_path):
        self.train_data_path = train_data_path

    def model_lgb(self, X, Y, process_num):
        # create dataset for lightgbm
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.1, random_state=0)  # 分训练集和验证集
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'cross_entropy'},
            'num_leaves': 31,
            'max_depth': 3,
            'learning_rate': 0.1,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 0
        }
        # train
        print("Training lgb model....")
        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval, early_stopping_rounds=10)
        print("Save model to " + process_num + ".joblib")
        dump(gbm, "models/" + process_num + ".joblib")

    def train_ents(self):
        train_data = open(self.train_data_path,'r',encoding='utf-8').readlines()
        fea_ents = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                '../coreEntityEmotion_baseline/models/stopwords.txt')
        X = []
        Y = []
        for news in tqdm(train_data):
            news = json.loads(news)
            X_data = fea_ents.combine_features(news)
            Y_data = [x['entity'] for x in news['coreEntityEmotions']]
            for x in X_data:
                if x[0][0] in Y_data:
                    Y.append(1)
                else:
                    Y.append(0)
                X.append(x[1])
        print("Save features... ")
        dump(X, "models/x1_featrues.joblib")
        dump(Y, "models/y1_featrues.joblib")
        # X = load("models/x1_featrues.joblib")
        # Y = load("models/y1_featrues.joblib")
        self.model_lgb(X, Y, "model1")
        print("done!")


if __name__ == "__main__":
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt')
    train.train_ents()