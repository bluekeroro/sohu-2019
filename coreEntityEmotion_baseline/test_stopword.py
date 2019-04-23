# -*- coding:UTF-8 -*-
"""
@File    : test_stopword.py
@Time    : 2019/4/23 14:57
@Author  : Blue Keroro
"""
import codecs
import json


def loadData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f.readlines():
        news = json.loads(line.strip())
        data.append(news)
    return data

if __name__ == '__main__':
    data = loadData('data/coreEntityEmotion_train.txt')
    stopword = set()
    with open('models/stopwords.txt','r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            stopword.add(line)
    cnt=0
    for news in data:
        entities = [x['entity'] for x in news['coreEntityEmotions']]
        for entitiy in entities:
            if entitiy in stopword:
                print(entitiy)
                stopword.remove(entitiy)
                cnt+=1

    print('number:',cnt)
    with open('models/stopwords.txt', 'w', encoding='utf-8') as file:
        for line in stopword:
            file.write(line+'\n')
