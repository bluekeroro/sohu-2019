# -*- coding:UTF-8 -*-
"""
@File    : test.py
@Time    : 2019/4/21 19:37
@Author  : Blue Keroro
"""
import csv
import time
import json
from joblib import load, dump
from tqdm import tqdm
import re
from sprint6.features_ents import feature_ents


class Test():
    def __init__(self, test_file, output_file, ents_model_path, feature_ents_func, debug=False):
        # 测试文件
        self.test_file = test_file
        self.output_file = output_file
        # 训练好的模型地址
        self.ents_model = load(ents_model_path)
        self.feature_ents_func = feature_ents_func

        self.debug = debug

    def test(self):
        test_file = open(self.test_file, 'r', encoding='utf-8').readlines()
        res_file = open(self.output_file, 'w', encoding='utf-8')
        # fea_ents = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
        #                         '../coreEntityEmotion_baseline/models/stopwords.txt')
        if self.debug is True:
            test_file = test_file[:int(len(test_file) / 10)]
        for news in tqdm(test_file):
            news = json.loads(news)
            ent_fea = self.feature_ents_func.combine_features(news)
            # 预测实体
            ent_predict_result = []
            for fea in ent_fea:
                ent_score = self.ents_model.predict([fea[1]])
                ent_predict_result.append([fea[0][0], ent_score])

            ent_predict_result.sort(key=lambda x: x[1], reverse=True)

            # 选前三个实体
            ents = [self.delete_mark(entity[0]) for entity in ent_predict_result[:3]]
            emos = ['POS' for i in ents[:3]]
            res_file.write('{}\t{}\t{}\n'.format(news['newsId'], ','.join(ents), ','.join(emos)))
        print("done")

    def delete_mark(self, str_input):
        # 处理书名号
        if str_input[0] == '《' and str_input[-1] == '》':
            return str_input[1:-1]
        if '《' in str_input and '》' not in str_input:
            str_input = str_input.replace('《', '')
        elif '》' in str_input and '《' not in str_input:
            str_input = str_input.replace('》', '')
        str_input = str_input.replace('\'', '').replace('\"', '').replace(',', '')  # 如果实体中只含有一个引号，会导致提交报错
        return str_input


if __name__ == '__main__':
    fea_ents = feature_ents('../coreEntityEmotion_baseline/models/nerDict.txt',
                            '../coreEntityEmotion_baseline/models/stopwords.txt')
    test = Test('../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1.txt',
                '../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1_result.txt',
                'models/model1.joblib', fea_ents)
    test.test()