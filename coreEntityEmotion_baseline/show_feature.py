# -*- coding:UTF-8 -*-
"""
@File    : show_feature.py
@Time    : 2019/4/17 15:03
@Author  : Blue Keroro
"""
import codecs
import json

from tqdm import tqdm


def loadNerDict():
    nerDictFile = codecs.open('models/nerDict.txt', 'r', 'utf-8')
    nerSet = set()
    for line in nerDictFile:
        nerSet.add(line.strip())
    return nerSet


def loadTrueData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = {}
    for line in f.readlines():
        news = json.loads(line.strip())
        data[news['newsId']] = {}
        data[news['newsId']]['text'] = news['title'] + '\n' + news['content']
        data[news['newsId']]['entity'] = []
        data[news['newsId']]['emotion'] = []
        for coreEntityEmotion in news['coreEntityEmotions']:
            data[news['newsId']]['entity'].append(coreEntityEmotion['entity'])
            data[news['newsId']]['emotion'].append(coreEntityEmotion['emotion'])
    return data


def loadPredData(filePath):
    data = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip().split("\t")
                data[line[0]] = {}
                data[line[0]]['entity'] = line[1].split(',')
                data[line[0]]['emotion'] = line[2].split(',')
            except IndexError:
                pass
    return data


if __name__ == '__main__':
    print('start')
    data = loadTrueData('data/coreEntityEmotion_train.txt')
    # data = loadPredData('data/coreEntityEmotion_sample_submission_v2.txt')
    nerSet = loadNerDict()
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cntSum = 0
    entity_set = set()
    cnt7 = 0
    errorDataStr = ''
    cntPOS = 0
    cntNORM = 0
    cntNEG = 0
    for newsId in tqdm(data):
        if 'entity' not in data[newsId]:
            print(newsId)
            continue
        if len(data[newsId]['entity']) < 3:
            # print(newsId)
            cnt += 1
            if len(data[newsId]['entity']) < 2:
                cnt5 += 1
        if len(data[newsId]['entity']) > 3:
            cnt2 += 1
        cntSum += len(data[newsId]['entity'])
        for entity in data[newsId]['entity']:
            entity_set.add(entity)
            tempSet = set()
            if '\'' in entity or '\"' in entity:
                cnt7 += 1
            if ' ' in entity:
                # print(entity)
                cnt4 += 1
            if entity not in nerSet:
                cnt3 += 1
                # tempSet.add(entity)
            if len(entity) < 2:
                cnt1 += 1
                # tempSet.add(entity)
            if len(entity) > 4:
                pass
                # tempSet.add(entity)
            if len(tempSet) > 0:
                pass
                # errorDataStr=errorDataStr+newsId+'\t'+','.join(tempSet)+'\n'
            if entity not in data[newsId]['text']:
                cnt6 += 1
        for emotion in data[newsId]['emotion']:
            if emotion == 'NEG':
                cntNEG += 1
            if emotion == 'NORM':
                cntNORM += 1
            if emotion == 'POS':
                cntPOS += 1

    # print(errorDataStr)
    print('实体超过三个的数量：', cnt2)
    print('实体不足三个的数量：', cnt)
    print('实体不足两个的数量：', cnt5)
    print('实体长度为一的数量：', cnt1)
    print('实体不在nerDict.txt中的数量：', cnt3)
    print('包含空格的实体的数量：', cnt4)
    print('训练集实体总数：', cntSum)
    print('实体不在文章的百分比：{}%'.format(cnt6 / cntSum * 100))
    print('含有非法引号的实体的个数：', cnt7)
    print('实体重复率：', (cntSum - len(entity_set)) / cntSum * 100, '%')
    print("消极实体百分比{}%".format(cntNEG / cntSum * 100))
    print("中立实体百分比{}%".format(cntNORM / cntSum * 100))
    print("积极实体百分比{}%".format(cntPOS / cntSum * 100))
    print('end')
    entity_list = []
    for newsId in tqdm(data):
        entity_list += data[newsId]['entity']
    entity_set = set(entity_list)
    tmp_list = [[entity, entity_list.count(entity)] for entity in entity_set]
    tmp_list.sort(key=lambda x: x[1],reverse=True)
    print(tmp_list)
