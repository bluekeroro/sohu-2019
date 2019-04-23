# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
import xgboost as xgb
import time
import json
from joblib import load, dump
from tqdm import tqdm
import re
from sprint6.features_ents import feature_ents
from sklearn.model_selection import train_test_split


class Train():
    def __init__(self, train_data_path, model_path, feature_ents_func, debug=False):
        self.train_data_path = train_data_path
        self.model_path = model_path
        self.feature_ents_func = feature_ents_func
        self.debug = debug

    def model_xgb(self, X, Y):
        # create dataset for lightgbm
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.1, random_state=0)  # 分训练集和验证集
        xgb_train = xgb.DMatrix(train_x, label=train_y)
        xgb_eval = xgb.DMatrix(valid_x, label=valid_y)

        # specify your configurations as a dict
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': 10,  # 类数，与 multisoftmax 并用
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 20,
            'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            'eta': 0.4,
            'eval_metric': 'merror',
            'silent': 0,
            'seed': 1000,
            'nthread': 4,  # cpu 线程数
        }
        # dtrain, dtest = Get_data()
        watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]
        # train
        print("Training xgb model....")
        evals_result = {}
        gbm = xgb.train(params, xgb_train, evals=watchlist, evals_result=evals_result, num_boost_round=200,
                        early_stopping_rounds=10)
        print('xgb 训练结果 evals_result：', evals_result)
        print("Save model to " + self.model_path)
        dump(gbm, self.model_path)

    def train_ents(self):
        train_data = open(self.train_data_path, 'r', encoding='utf-8').readlines()
        if self.debug is True:
            train_data = train_data[:int(len(train_data) / 10)]
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
        # X = load("models/x1_featrues.joblib")f
        # Y = load("models/y1_featrues.joblib")
        self.model_xgb(X, Y)
        print("done!")


if __name__ == "__main__":
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model1.joblib', feature_ents_func)
    train.train_ents()
