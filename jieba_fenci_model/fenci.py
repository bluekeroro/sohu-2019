# -*- coding:UTF-8 -*-
"""
@File    : fenci.py
@Time    : 2019/4/26 15:52
@Author  : Blue Keroro
"""
import json

import jieba
import jieba.analyse
from tqdm import tqdm

from jieba_fenci_model.features_ents import feature_ents


def save_fenci_feature_func(input_file, output_file, feature_ents_func):
    '''
    该方法不会改变原有的output_file里的数据。如需重新生成全部数据，请手动删除output_file
    :param input_file:
    :param output_file:
    :param feature_ents_func:
    :return:
    '''
    train_data = list()
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            train_data.append(line)

    result_set = set()
    with open(output_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = eval(line)
            result_set |= set(line)

    result = open(output_file, 'a', encoding='utf-8')
    for news in tqdm(train_data):
        news = json.loads(news)
        if news['newsId'] in result_set:
            continue
        content_words_tfidf, title_words_tfidf = feature_ents_func.get_tfidf_Score(news)
        content_words_textRank, title_words_textRank = feature_ents_func.get_textRank_Score(news)
        keys = content_words_tfidf.keys() | title_words_tfidf.keys() | content_words_textRank.keys() | title_words_textRank.keys()
        ner_dict = {}
        ner_dict[news['newsId']] = {}
        for ner in keys:
            features = [content_words_tfidf[ner] if ner in content_words_tfidf else 0,
                        title_words_tfidf[ner] if ner in title_words_tfidf else 0,
                        content_words_textRank[ner] if ner in content_words_textRank else 0,
                        title_words_textRank[ner] if ner in title_words_textRank else 0,
                        (feature_ents_func.key_word_pos.index(feature_ents_func.word_pos[ner])
                         if (ner in feature_ents_func.word_pos
                             and feature_ents_func.word_pos[ner] in feature_ents_func.key_word_pos)
                         else feature_ents_func.key_word_pos.index('n')) * 0.1]  # 可增加新的特征
            ner_dict[news['newsId']][ner] = features
        result.write(str(ner_dict) + '\n')
    result.close()


def get_fenci_feature_func(input_file):
    data_dict = dict()
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = eval(line)
            for key in line:
                data_dict[key] = line[key]
    return data_dict


if __name__ == '__main__':
    feature_ents_func = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                                     '../coreEntityEmotion_baseline/models/stopwords.txt')
    # save_fenci_feature_func('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt',
    #                         'result/result_jieba_fenci.txt',
    #                         feature_ents_func)
    save_fenci_feature_func('../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1.txt',
                            'result/result_jieba_fenci.txt',
                            feature_ents_func)
