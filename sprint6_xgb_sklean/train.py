# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/4/21 18:49
@Author  : Blue Keroro
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
import json
from joblib import load, dump
from tqdm import tqdm
from sprint6_xgb_sklean.features_ents import feature_ents
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, ShuffleSplit, \
    StratifiedKFold
from xgboost.sklearn import XGBRegressor


class Train():
    def __init__(self, train_data_path, model_path, feature_ents_func, debug=False):
        self.train_data_path = train_data_path
        self.model_path = model_path
        self.feature_ents_func = feature_ents_func
        self.debug = debug

    # params = {'booster': 'gbtree', 'eta': 0.138, 'max_depth': 2, 'n_estimators': 100, 'silent': 0,
    #           'objective': 'binary:logistic'}  # 0.439
    def model_xgb(self, X, Y):
        xgb_model = XGBRegressor()
        print('model_xgb fit')
        gbm = xgb_model.fit(X, Y)
        print("feature_importances_ : ", gbm.feature_importances_)
        print("Save model to " + self.model_path)
        dump(gbm, self.model_path)

    def model_xgb_search(self, X, Y):
        # train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.1, random_state=0)  # 分训练集和验证集
        print('model_xgb_search start')
        xgb_model = XGBRegressor(nthread=4)

        cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.2)
        # param_grid = dict(
        #     max_depth=[2],
        #     min_child_weight= [1, 2, 3, 4, 5, 6],
        #     gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        #     learning_rate=np.linspace(0.03, 1, 10),
        #     n_estimators=[50, 100, 200, 400],
        #     num_class=[2],
        #     objective=['multi:softmax']
        # )
        param_grid = dict(
            max_depth=[1, 2, 3],
            # learning_rate=np.linspace(0.03, 0.3, 5),
            n_estimators=[100, 200],
            num_class=[2],
            objective=['multi:softmax']  # 'binary:logistic'
        )
        start = time.time()
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        grid = GridSearchCV(xgb_model, param_grid, cv=cv_split)  # scoring='neg_log_loss'
        grid_result = grid.fit(X, Y)
        print("Best: %f using params: %s estimator: %s" % (
            grid_result.best_score_, grid_result.best_params_, grid_result.best_estimator_))
        print('GridSearchCV process use %.2f seconds' % (time.time() - start))
        print('end=======')

    def train_ents(self, load_feature_model=False):
        X = []
        Y = []
        if load_feature_model is False:
            train_data = list()
            with open(self.train_data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    train_data.append(line)
            if self.debug is True:
                train_data = train_data[:int(len(train_data) / 10)]
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
            print("Save features end... ")
        else:
            X = load("models/x1_featrues.joblib")
            Y = load("models/y1_featrues.joblib")
        # X = load("models/x1_featrues.joblib")
        # Y = load("models/y1_featrues.joblib")
        self.model_xgb(X, Y)
        # self.model_xgb_tmp(X, Y)
        # self.model_xgb_search(X, Y)
        print("done!")


if __name__ == "__main__":
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    train = Train('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
                  'models/model1.joblib', feature_ents_func)
    train.train_ents()
