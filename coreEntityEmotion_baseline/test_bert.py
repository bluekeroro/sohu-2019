# -*- coding:UTF-8 -*-
"""
@File    : test_bert.py
@Time    : 2019/5/9 0:14
@Author  : Blue Keroro
"""
import codecs
import json

from tqdm import tqdm

from bert_reduce import loadData


def loadData_(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f.readlines():
        news = json.loads(line.strip())
        data.append(news)
    return data


if __name__ == '__main__':
    bert_data = loadData('data/result_bert.txt')
    train_data = loadData_('data/coreEntityEmotion_train.txt')
    cnt = 0
    cntSum = 0
    for news in tqdm(train_data):
        cntSum += len(news['coreEntityEmotions'])
        for coreEntityEmotion in news['coreEntityEmotions']:
            entity = coreEntityEmotion['entity']
            if news['newsId'] in bert_data and entity in bert_data[news['newsId']]:
                cnt += 1
    print('bert 准确率：{}%'.format(cnt / cntSum * 100))
