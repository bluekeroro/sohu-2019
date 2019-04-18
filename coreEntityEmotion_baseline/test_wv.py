# -*- coding:UTF-8 -*-
"""
@File    : test_wv.py
@Time    : 2019/4/17 22:21
@Author  : Blue Keroro
"""

from gensim.models.word2vec import Word2VecKeyedVectors

if __name__ == '__main__':
    wv_from_text = Word2VecKeyedVectors.load_word2vec_format('C:/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt', binary=False)
    print(wv_from_text.word_vec('因吹斯汀'))
    print('')