# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
import time
import json
from joblib import load, dump
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import re
from sprint_stacking.features_ents import feature_ents


class Train():
    def __init__(self, train_data_path, model_path, feature_ents_func, debug=False):
        self.train_data_path = train_data_path
        self.model_path = model_path
        self.feature_ents_func = feature_ents_func
        self.debug = debug


    def train_ents(self):
        # train_data = open(self.train_data_path, 'r', encoding='utf-8').readlines()
        train_data = list()
        with open(self.train_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                train_data.append(line)

        if self.debug is True:
            train_data = train_data[:int(len(train_data) / 10 / 100)]
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
        if self.debug is False:
            return X,Y
        dump(X, "models/x1_featrues.joblib")
        dump(Y, "models/y1_featrues.joblib")
        print("done!")


if __name__ == "__main__":
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model1.joblib', feature_ents_func)
    train.train_ents()
