# -*- coding: utf-8 -*-
import codecs
import json

import jieba


class Train():
    def __init__(self):
        jieba.load_userdict('../coreEntityEmotion_baseline/models/nerDict.txt')

    def trainCoreEntity(self):
        pass

    def loadData(self, filePath):
        f = codecs.open(filePath, 'r', 'utf-8')
        data = []
        for line in f.readlines():
            news = json.loads(line.strip())
            data.append(news)
        return data

if __name__ == '__main__':
    pass