# -*- coding:UTF-8 -*-
"""
@File    : test_tfidf.py
@Time    : 2019/4/18 0:25
@Author  : Blue Keroro
"""
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    document = ['我 想 看 爱，死亡与机器人','他 的']
    model = TfidfVectorizer().fit(document)
    print(model.get_feature_names())
    print('')