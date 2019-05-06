# -*- coding:UTF-8 -*-
"""
@File    : ltp_reduce.py
@Time    : 2019/5/5 1:21
@Author  : Blue Keroro
"""

import codecs
import json
import numpy as np
from tqdm import tqdm


# label: {'DIS', 'TPC', 'C-A0', 'FRQ', 'A3', 'PRP', 'MNR', 'A1', 'A0', 'EXT', 'A4', 'CND', 'A2', 'DIR', 'BNF', 'TMP', 'ADV', 'C-A1', 'LOC'}
# used {'A0','A1','LOC','PRP','TPC'}
def loadData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f:
        news = json.loads(line.strip())
        data.append(news)
    return data


class LTP(object):
    def __init__(self, ltp_result_path):
        self.ltp_result_path = ltp_result_path
        print('LTP __init__ start')
        data = loadData(ltp_result_path)
        self.data = {}
        for news in tqdm(data):
            self.data[news['newsId']] = news['label']
        print('LTP __init__ end')
        # 前5 重点
        self.labels = ['TPC', 'A0', 'A1', 'LOC', 'PRP', 'C-A0', 'C-A1', 'MNR', 'CND', 'A2', 'A3', 'A4', 'DIR', 'TMP',
                       'ADV', 'FRQ', 'DIS', 'BNF', 'EXT']

    def get_label(self, word, newsId, model=1):
        '''
        model 为1或2时返回list
        :param word:
        :param newsId:
        :param model:
        :return:
        '''
        one_hot_encoding = np.zeros(len(self.labels) + 1)
        if model == 1:
            for index, lable in enumerate(self.labels):
                if lable in self.data[newsId] and word in self.data[newsId][lable]:
                    one_hot_encoding[index] = 1
                    return list(one_hot_encoding)
            one_hot_encoding[len(self.labels)] = 1
            return list(one_hot_encoding)
        elif model == 2:
            flag = 0
            for index, lable in enumerate(self.labels):
                if lable in self.data[newsId] and word in self.data[newsId][lable]:
                    one_hot_encoding[index] = 1
                    flag = 1
            if flag == 0:
                one_hot_encoding[len(self.labels)] = 1
            return list(one_hot_encoding)
        elif model == 3:
            for index, lable in enumerate(self.labels):
                if lable in self.data[newsId] and word in self.data[newsId][lable]:
                    return index * 0.1
            return 0.1