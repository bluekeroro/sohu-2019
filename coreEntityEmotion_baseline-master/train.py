# -*- coding: utf-8 -*-

import jieba
import codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from joblib import dump
import re
from scipy.sparse import vstack
from tqdm import tqdm

class Train():
    def __init__(self):
        # load nerDict as named entity recognizer
        self.loadNerDict()

    def trainCoreEntity(self):
        '''
        train model for coreEntity
        Baseline use entityDict for named entity recognition, you can use a more wise method.
        Baseline use tfIdf score as feature and LR as classification model
        :return:
        '''
        # 1. train tfIdf as core entity score model
        trainData = self.loadData('data/coreEntityEmotion_train.txt')
        # trainData = trainData[0:10] # 减小数量
        print("loading all ner corpus from train data...")

        nerCorpus = []
        for news in tqdm(trainData):
            nerCorpus.append(' '.join(self.getEntity(news)))

        print("fitting ner tfIdf model...")
        tfIdf = TfidfVectorizer()
        tfIdf.fit(nerCorpus)
        # 1.1 save tfIdf model
        dump(tfIdf, 'models/nerTfIdf.joblib')

        # 2. train LR with tfIdf score as features
        isCoreX = []
        isCoreY = []
        for news in trainData:

            tfIdfNameScore = self.getTfIdfScore(news, tfIdf) # 获取到关键词与归一化处理词频相对应的list[(关键词,归一化后的词频)...]
            # 关键词经过nerDict过滤

            coreEntity_GroundTruth = [x['entity'] for x in news['coreEntityEmotions']]
            for name, score in tfIdfNameScore:
                if (name in coreEntity_GroundTruth):
                    isCoreX.append([score])
                    isCoreY.append(1)
                else:
                    isCoreX.append([score])
                    isCoreY.append(0)

        # 3. train LR model for coreEntity
        print("training LR model for coreEntity...")
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(isCoreX, isCoreY) # 将关键词词频和非关键词词频进行训练
        dump(clf, 'models/CoreEntityCLF.joblib')

    def trainEmotion(self):
        '''
        train emotion model
        Baseline use tfIdf vector as feature, linearSVC as classfication model
        :return:
        '''
        trainData = self.loadData('data/coreEntityEmotion_train.txt')
        # trainData = trainData[0:10] # 减小数量
        emotionX = []
        emotionY = []

        print("loading emotion corpus from train data...")

        # 1. get all related sentences to the entities
        for news in tqdm(trainData):

            text = news['title'] + '\n' + news['content']
            entities = [x['entity'] for x in news['coreEntityEmotions']]
            emotions = [x['emotion'] for x in news['coreEntityEmotions']]
            entityEmotionMap = dict(zip(entities, emotions)) # {关键词：情感,...}
            entitySentsMap = {}
            for entity in entityEmotionMap.keys():
                entitySentsMap[entity] = []

            for sent in re.split(r'[\n\t，。！？“”（）]', text): # 取出每篇文章的语句
                for entity in entityEmotionMap.keys():
                    if (entity in sent):
                        entitySentsMap[entity].append(sent)  # 收集含有对应关键词的语句

            for entity, sents in entitySentsMap.items():
                relatedText = ' '.join(sents)
                emotionX.append([relatedText])  # 关键词对应的语句
                emotionY.append(entityEmotionMap[entity])  # 关键词对应的情感

        # 2. train tf-idf model for emotion related words
        emotionWordCorpus = []
        for news in trainData:
            emotionWordCorpus.append(' '.join(self.getWords(news)))  # 收集每篇文章所有的关键词

        print("fitting emotion tfIdf model...")

        tfIdf = TfidfVectorizer()
        tfIdf.fit(emotionWordCorpus)
        dump(tfIdf, 'models/emotionTfIdf.joblib')

        # 3. use naive bayes to train emotion classifiction
        emotionX = vstack([tfIdf.transform(x) for x in emotionX]).toarray() # 拼接全部的词频矩阵

        print("training emotion clf with linearSVC...")

        print(emotionX.shape)
        clf = MultinomialNB()
        clf.fit(emotionX, emotionY)

        print(clf.score(emotionX, emotionY))

        dump(clf, 'models/emotionCLF.joblib')

    def getTfIdfScore(self, news, tfIdf):
        featureName = tfIdf.get_feature_names()  # 获得文本的关键词

        doc = self.getEntity(news)  # list  所有文本和nerDict匹配 过滤掉不在nerDict的实体

        tfIdfFeatures = tfIdf.transform([' '.join(doc)])    # 转换为一维词频向量，即tf-idf矩阵

        tfIdfScores = tfIdfFeatures.data  # 转换为n维一列词频矩阵
        # normalize
        tfIdfScoresNorm = normalize([tfIdfScores], norm='max') # 归一化处理

        tfIdfNameScore = [(featureName[x[0]], x[1]) for x in zip(tfIdfFeatures.indices, tfIdfScoresNorm[0])]  # 将关键词与归一化的词频对应起来
        tfIdfNameScore = sorted(tfIdfNameScore, key=lambda x: x[1], reverse=True)

        return tfIdfNameScore

    def loadNerDict(self):
        nerDictFile = codecs.open('models/nerDict.txt', 'r', 'utf-8')
        self.nerDict = []
        for line in nerDictFile:
            self.nerDict.append(line.strip())

    def getWords(self, news):
        '''
        get all word list from news
        :param news:
        :return:
        '''
        title = news['title']
        content = news['content']

        words = jieba.cut(title + '\t' + content)

        return list(words)

    def getEntity(self, news):
        '''
        get all entity list from news
        :param news:
        :return:
        '''
        ners = []
        words = self.getWords(news)
        for word in words:
            if (word in self.nerDict):
                ners.append(word)
        return ners

    def loadData(self, filePath):
        f = codecs.open(filePath, 'r', 'utf-8')
        data = []
        for line in f.readlines():
            news = json.loads(line.strip())
            data.append(news)
        return data


if __name__ == '__main__':
    trainer = Train()
    trainer.trainCoreEntity()
    trainer.trainEmotion()
