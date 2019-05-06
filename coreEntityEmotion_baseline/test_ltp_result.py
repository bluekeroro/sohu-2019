# -*- coding:UTF-8 -*-
"""
@File    : test_ltp_result.py
@Time    : 2019/5/4 22:42
@Author  : Blue Keroro
"""
import codecs
import json

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


if __name__ == '__main__':
    data = loadData('data/result_ltp.txt')
    label = set()
    for news in tqdm(data):
        label |= news['label'].keys()
    print(label)
