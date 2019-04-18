# -*- coding: utf-8 -*-
from joblib import load
from coreEntityEmotion_baseline.train import Train
import re
import codecs
from tqdm import tqdm

class Test(Train):
    def __init__(self):
        super(Test, self).__init__()
        self.coreEntityTfIdf = load('coreEntityEmotion_baseline/models/nerTfIdf.joblib')
        self.coreEntityCLF = load('coreEntityEmotion_baseline/models/CoreEntityCLF.joblib')

        # self.emotionTfIdf = load('models/emotionTfIdf.joblib')
        # self.emotionCLF = load('models/emotionCLF.joblib')

    def testCoreEntity(self):
        testData = self.loadData('coreEntityEmotion_baseline/data/2_coreEntityEmotion_train.txt')
        testData = testData[:100]

        f_submit = codecs.open('coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt',
                                        'w', 'utf-8')

        for news in tqdm(testData):
            # print(news)
            predictCoreEntityEmotion = {}

            tfIdfNameScore = self.getTfIdfScore(news, self.coreEntityTfIdf)

            # predict core Entities
            coreEntities = []
            for name, score in tfIdfNameScore:
                # print(self.coreEntityCLF.predict([[score]]))
                if(self.coreEntityCLF.predict([[score]]) > 0.5):
                    coreEntities.append(name)
            # print('newsId:',news['newsId'],'coreEntities:',coreEntities)
            # predict emotion of core entity
            for entity in coreEntities:
                text = news['title'] + '\n' + news['content']
                relatedSents = []
                for sent in re.split(r'[\n\t，。！？“”（）]', text):
                    if (entity in sent):
                        relatedSents.append(sent)
                # relatedText = ' '.join(relatedSents)
                # emotionTfIdfFeature = self.emotionTfIdf.transform([relatedText]).toarray()
                # emotion = self.emotionCLF.predict(emotionTfIdfFeature)
                predictCoreEntityEmotion[entity] ='POS'    #emotion[0]

            all_entities = []
            all_emotions = []
            for entity, emotion in predictCoreEntityEmotion.items():
                all_entities.append(entity.replace('\t', '').replace('\n', '').replace(',', ''))
                all_emotions.append(emotion)

            f_submit.write(news['newsId'])
            f_submit.write('\t')
            f_submit.write(','.join(all_entities))
            f_submit.write('\t')
            f_submit.write(','.join(all_emotions))
            f_submit.write('\n')

if __name__ == '__main__':
    test = Test()
    test.testCoreEntity()


