# -*- coding:UTF-8 -*-
"""
@File    : test_jieba.py
@Time    : 2019/4/17 22:06
@Author  : Blue Keroro
"""
import jieba.analyse

if __name__ == '__main__':
    file = open('data/one_text.txt', 'r', encoding='utf-8')
    sentence = file.readlines()
    # jieba.load_userdict('models/nerDict.txt')
    words = jieba.analyse.extract_tags(sentence[0], topK=20, withWeight=False, allowPOS=())
    print(words)
