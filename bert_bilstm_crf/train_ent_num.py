# -*- coding:UTF-8 -*-
"""
@File    : train_ent_num.py
@Time    : 2019/5/5 22:51
@Author  : Blue Keroro
"""
import sys
import os

from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import codecs
import json
import random

from kashgari.tasks.classification import *
from tqdm import tqdm
import jieba
import numpy as np


def loadData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f:
        news = json.loads(line.strip())
        data.append(news)
    return data


def read_data_file(path):
    data = loadData(path)
    # data = data[:20000]
    random.shuffle(data)
    x_list = []
    y_list = []
    for news in tqdm(data):
        y_list.append(length_to_label(len(news['coreEntityEmotions'])))
        x_list.append(reduce_text(news))
    return x_list, y_list


def reduce_text(news):
    text = news['title'] + '。' + news['content']
    text = text.replace('\n', '').replace('\t', '')
    return list(jieba.cut(text))
    # return [list(jieba.cut(news['title'].replace('\n', '').replace('\t', ''))),
    #         list(jieba.cut(news['content'].replace('\n', '').replace('\t', '')))]
    # 可将title content分开 报错


def length_to_label(length):
    if length > 3:
        length = 3
    return str(length)
    # label = [0,0,0,0]  # 0,1,2,3
    # if length >= 3:
    #     label[3] = 1
    # else:
    #     label[length] = 1
    # return label


# def label_to_lengt(label):
#     return list(label).index(1)


def train(train_x, train_y):
    embedding = BERTEmbedding("bert-base-chinese", sequence_length=512)
    model = BLSTMModel(embedding)
    # model = CNNModel(embedding)

    # tmp = []
    # for i in train_x:
    #     tmp.append([i])
    # train_x = tmp
    length = int(len(train_x) * 0.9)
    print(len(train_x[:length]), len(train_y[:length]))
    model.fit(train_x[:length], train_y[:length], train_x[length:], train_y[length:])
    model.save('BLSTM_model')


if __name__ == '__main__':
    train_x, train_y = read_data_file('data/coreEntityEmotion_train.txt')
    train(train_x, train_y)
    # model = CNNModel().load_model('cnn_model')
    model = BLSTMModel.load_model('BLSTM_model')
    data = loadData('data/coreEntityEmotion_test_stage1.txt')
    newsId_set = set()
    try:
        with open("data/result_ent_num.txt", 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = line.split('\t')
                newsId_set.add(line[0])
    except IOError:
        print('该文件不存在')

    with open("data/result_ent_num.txt", 'a', encoding='utf-8') as file:
        for news in tqdm(data):
            if news['newsId'] in newsId_set:
                continue
            test_x = reduce_text(news)
            label = model.predict([test_x])[0]
            # length = label_to_lengt(label)
            file.write('{}\t{}\n'.format(news['newsId'], label))
