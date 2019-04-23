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
    # words = jieba.analyse.extract_tags(sentence[0], topK=20, withWeight=False, allowPOS=())
    # print(words)
    # key_word_pos = ('x', 'ns', 'n', 'vn', 'v', 'l', 'j', 'nr', 'nrt', 'nt', 'nz', 'nrfg', 'm', 'i', 'an', 'f', 't',
    #                 'b', 'a', 'd', 'q', 's', 'z')
    # key_word_pos = ('ns', 'n', 'vn', 'v','nr')
    key_word_pos = ('ns', 'n', 'vn', 'v', 'l', 'j', 'nr', 'nrt', 'nt', 'nz', 'nrfg', 'an','s')
    print(sentence[0])
    words1 = jieba.analyse.extract_tags(sentence[0], topK=20, withWeight=True, allowPOS=key_word_pos)
    words2 = jieba.analyse.textrank(sentence[0], topK=20, withWeight=True, allowPOS=key_word_pos)
    print(words1)
    print(words2)
