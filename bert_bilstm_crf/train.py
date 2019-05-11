# -*- coding:UTF-8 -*-
"""
@File    : train.py
@Time    : 2019/5/4 18:13
@Author  : Blue Keroro
"""
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from tqdm import tqdm
from time import time
from sklearn.utils import class_weight
from data_reduce import get_train_data, predict_reduce, loadData
import numpy as np


def reduce_text(news):
    text = news['title'] + '。' + news['content']
    text = text.replace('\n', '').replace('\t', '')
    return list(text)


if __name__ == '__main__':
    start = time()
    print('train start')
    train_x, train_y = get_train_data('data/train_text.txt')
    embedding = BERTEmbedding("bert-base-chinese", sequence_length=512)
    model = BLSTMCRFModel(embedding)
    length = int(len(train_x) * 0.9)
    print(len(train_x[:length]), len(train_y[:length]))
    model.fit(train_x[:length], train_y[:length], train_x[length:], train_y[length:], epochs=5, batch_size=20)
    # model.fit(train_x[:length], train_y[:length], train_x[length:], train_y[length:], epochs=5, batch_size=128,
    #           labels_weight=True, default_labels_weight=100)
    valid_x = train_x[length:]
    valid_y = train_y[length:]
    model.save('models')
    print('train end')
    print('predict start')
    try:
        model = BLSTMCRFModel.load_model('models')
    except Exception:
        print('模型加载失败')
    newsId_set = set()
    try:
        with open('data/result_bert.txt', 'r', encoding='utf-8') as file:
            for line in file:
                newsId_set.add(line.split('\t')[0])
    except IOError:
        print('文件不存在')

    test_data = loadData('data/coreEntityEmotion_test_stage1.txt')
    test_data += loadData('data/coreEntityEmotion_train.txt')
    with open('data/result_bert.txt', 'a', encoding='utf-8') as file:
        for news in tqdm(test_data):
            if news['newsId'] in newsId_set:
                continue
            test = reduce_text(news)
            pred_y = model.predict([test])[0]
            entitys = predict_reduce(test, pred_y)
            file.write('{}\t{}\n'.format(news['newsId'], ','.join(entitys)))
    print('save complete')
    cnt = 0
    cntSum = 0
    for i in tqdm(range(len(valid_x))):
        pred_y = model.predict([valid_x[i]])[0]
        entitys = predict_reduce(valid_x[i], pred_y)
        valid_entitys = predict_reduce(valid_x[i], valid_y[i])
        cntSum += len(valid_entitys)
        for i in entitys:
            if i in valid_entitys:
                cnt += 1
    print('bert 准确率：{}%'.format(cnt / cntSum * 100))
    print('used time:', time() - start)
