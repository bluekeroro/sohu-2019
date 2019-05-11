# -*- coding:UTF-8 -*-
"""
@File    : data_reduce.py
@Time    : 2019/5/4 18:13
@Author  : Blue Keroro
"""
import codecs
import json
import random

from tqdm import tqdm


def loadData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f:
        news = json.loads(line.strip())
        data.append(news)
    return data


def reduce(news, label):
    text = news['title'] + '。' + news['content']
    text = text.replace('\n', '').replace('\t', '')
    entitys = []
    for coreEntityEmotion in news['coreEntityEmotions']:
        entitys.append(coreEntityEmotion['entity'])
    entitys = sorted(entitys, key=lambda x: len(x))
    label_map = [label[0] for i in text]
    for entity in entitys:
        index = text.find(entity, 0)
        while index != -1:
            for i in range(len(entity)):
                label_map[index + i] = (label[1] if i == 0 else label[2])
            index = text.find(entity, index + 1)
    ret_list = []
    for index in range(len(text)):
        ret_list.append(text[index] + ' ' + label_map[index])
    return ret_list


def get_train_data(data_path):
    train_data = []
    tmp = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line == '\n':
                train_data.append(tmp)
                tmp = []
            else:
                tmp.append(line.strip('\n'))
    random.shuffle(train_data)
    train_data = [[j.split(' ') for j in i] for i in train_data]
    train_x = [[token[0] for token in sen] for sen in train_data]
    train_y = [[token[1] for token in sen] for sen in train_data]
    return train_x, train_y


def predict_reduce(test_x, pred_y):
    index = -1
    entitys = set()
    while (1):
        try:
            index = pred_y.index("B-ENT", index + 1)
            entity = test_x[index]
            for i in range(index + 1, len(test_x)):
                if pred_y[i] != "I-ENT":
                    break
                entity += test_x[i]
            entitys.add(entity)
        except Exception:
            break
    return entitys


if __name__ == '__main__':
    trainData = loadData('data/coreEntityEmotion_train.txt')
    # trainData = trainData[:2000]
    label = ["O", "B-ENT", "I-ENT"]
    with open('data/train_text.txt', 'w', encoding='utf-8') as file:
        for news in tqdm(trainData):
            data = reduce(news, label)
            file.write('\n'.join(data))
            file.write('\n\n')

    # train_x, train_y = get_train_data('data/train_text.txt')
    # print('train_x', train_x)
    # print('train_y', train_y)
    # print(predict_reduce(train_x[0], train_y[0]))
