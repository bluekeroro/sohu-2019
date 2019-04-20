# -*- coding:UTF-8 -*-
"""
@File    : test_wv.py
@Time    : 2019/4/17 22:21
@Author  : Blue Keroro
"""

from gensim.models.word2vec import Word2VecKeyedVectors

if __name__ == '__main__':
    from time import time

    start = time()
    print('加载词向量')
    wv_from_text = Word2VecKeyedVectors.load_word2vec_format(
        'C:/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt', binary=False)
    print('加载词向量    完毕')
    with open('models/nerDict.txt', 'r', encoding='utf-8') as f1, \
            open('models/nerDict_tencent_error.txt', 'w', encoding='utf-8') as f2:
        for line in f1:
            line = line.strip()
            try:
                if line not in wv_from_text:
                    f2.write(line + '\n')
            except Exception as e:
                print('出现问题', 'line:', line, 'error:', e)

    print('end', time() - start)
