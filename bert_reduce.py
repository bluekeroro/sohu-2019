# -*- coding:UTF-8 -*-
"""
@File    : bert_reduce.py
@Time    : 2019/5/6 21:33
@Author  : Blue Keroro
"""
from tqdm import tqdm


def loadData(bert_result_path):
    data = {}
    with open(bert_result_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = line.split('\t')
            if len(line)<2:
                continue
            data[line[0]] = line[1].split(',')
    return data


class bert(object):
    def __init__(self, bert_result_path):
        print('BERT __init__ start')
        self.data = loadData(bert_result_path)
        print('BERT __init__ end')

    def is_in_bert(self, newsId, word):
        if newsId in self.data and word in self.data[newsId]:
            return 1
        else:
            return 0


if __name__ == '__main__':
    obj = bert('coreEntityEmotion_baseline/data/result_bert.txt')
    print(obj.is_in_bert('bb856a3f', '英国2'))
