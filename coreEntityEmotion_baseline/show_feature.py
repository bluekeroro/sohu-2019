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
    cnt=0
    cnt1=0
    cnt2=0
    cnt3 =0
    cnt4 =0
    errorDataStr=''
    for newsId in tqdm(data):
        if 'entity' not in data[newsId]:
            print(newsId)
            continue
        if len(data[newsId]['entity'])<3:
            # print(newsId)
            cnt+=1
        if len(data[newsId]['entity'])>3:
            cnt2+=1
        for entity in data[newsId]['entity']:
            tempSet = set()
            if ' ' in entity:
                # print(entity)
                cnt4+=1
            if entity not in nerSet:
                cnt3+=1
                # tempSet.add(entity)
            if len(entity)<2:
                cnt1+=1
                tempSet.add(entity)
            if len(entity)>4:
                tempSet.add(entity)
            if len(tempSet)>0:
                errorDataStr=errorDataStr+newsId+'\t'+','.join(tempSet)+'\n'
    print(errorDataStr)
    print('实体超过三个的数量：', cnt2)
    print('实体不足三个的数量：',cnt)
    print('实体长度为一的数量：',cnt1)
    print('实体不在nerDict.txt中的数量：',cnt3)
    print('包含空格的实体的数量：', cnt4)
    print('end')