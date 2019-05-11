# -*- coding:UTF-8 -*-
"""
@File    : stacking.py
@Time    : 2019/4/28 19:50
@Author  : Blue Keroro
"""
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


def predict(X_train, Y_train, X_test):
    print("Y_train is 1:", Y_train.count(1))
    print("Y_train is 0:", Y_train.count(0))
    clfs = [
        LGBMRegressor(learning_rate=0.0475, max_depth=13, n_estimators=100, num_leaves=80),
        XGBRegressor(learning_rate=0.0475, max_depth=4, n_estimators=300)]
    X = np.array(X_train, dtype='float32')
    y = np.array(Y_train, dtype='float32')
    X_predict = np.array(X_test, dtype='float32')
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)), dtype='float32')
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)), dtype='float32')

    '''5折stacking'''
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds)
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        print("clf", j)
        dataset_blend_test_j = np.zeros((X_predict.shape[0], n_folds), dtype='float32')
        for i, (train, test) in enumerate(skf.split(X, y)):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            print("stacking Fold", i)
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            # if j == 0:
            #     class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
            #     clf.class_weight = dict(enumerate(class_weights))
            # else:
            #     class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
            #     clf.scale_pos_weight = class_weights[1] / class_weights[0]
            #     print('scale_pos_weight:', clf.scale_pos_weight)
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_predict)
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        del dataset_blend_test_j
        # print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
    # clf = LogisticRegression()
    # clf = GradientBoostingRegressor(learning_rate=0.02, max_depth=6)
    clf = LGBMRegressor()
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    clf.class_weight = dict(enumerate(class_weights))
    dataset_blend_train = np.append(dataset_blend_train, X, axis=1)
    dataset_blend_test = np.append(dataset_blend_test, X_predict, axis=1)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)
    return y_submission


if __name__ == '__main__':
    pass
