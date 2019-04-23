# -*- coding:UTF-8 -*-
"""
@File    : test_pkuseg.py
@Time    : 2019/4/23 15:17
@Author  : Blue Keroro
"""
import codecs
import json

import pkuseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from tqdm import tqdm


def loadData(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f.readlines():
        news = json.loads(line.strip())
        data.append(news)
    return data


if __name__ == '__main__':
    stopwordlist = []
    with open('models/stopwords.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            stopwordlist.append(line)
    # userList = []
    # with open('models/nerDict.txt', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         line = line.strip()
    #         userList.append(line)
    pkus = pkuseg.pkuseg(model_name='pkuseg_models/web')
    data = loadData('data/coreEntityEmotion_train.txt')
    data = data[:3200]
    cnt = 0
    cntSum = 0
    for news in tqdm(data):
        entities = [x['entity'] for x in news['coreEntityEmotions']]
        cntSum += len(entities)
        words = [' '.join(list(pkus.cut(news['title'] + '\t' + news['content'])))]
        vectorizer = CountVectorizer(stop_words=stopwordlist)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(words))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        feature_names = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()
        word_tfidf_dict = []
        for j in range(len(feature_names)):
            # print(feature_names[j], weight[0][j])
            word_tfidf_dict.append([feature_names[j], weight[0][j]])
        word_tfidf_dict = sorted(word_tfidf_dict, key=lambda x: x[1], reverse=True)

        for x in word_tfidf_dict[:40]:
            if x[0] in entities:
                cnt += 1
    print("pkuseg 分词准确率：{}%".format(cnt / cntSum * 100))

    # 未加stopWord  63.493182886694875%
    # 加stopWord   64.51574988246357%
    # 前40 62.89374706158909%
    # 增加用户词典 43.08885754583921%
    # 去除用户词典 增加news模型 60.836859426422194%
    # 增加用户词典 增加news模型 43.17113305124589%
    # 去除用户词典 增加web模型 64.56276445698167%
    # token_pattern=r"(?u)\b\w+\b" 64.03385049365303%
