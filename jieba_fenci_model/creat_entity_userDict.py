# -*- coding:UTF-8 -*-
"""
@File    : creat_entity_userDict.py
@Time    : 2019/4/29 17:53
@Author  : Blue Keroro
"""
from f1_score import loadTrueData

if __name__ == '__main__':
    trainData = loadTrueData('../coreEntityEmotion_baseline/data/coreEntityEmotion_train.txt')
    entity_set = set()
    for newsId in trainData:
        for entity in trainData[newsId]['entity']:
            entity_set.add(entity)
    with open('result/entity_userDict_pos.txt', 'w', encoding='utf-8') as file1, open('result/entity_userDict.txt', 'w',
                                                                                      encoding='utf-8') as file2:
        for entity in entity_set:
            file1.write('{} {} {}\n'.format(entity, 3, 'n'))
            file2.write(entity + '\n')

    # entity_set = set()
    # with open('../coreEntityEmotion_baseline/models/nerDict.txt', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         line = line.strip()
    #         entity_set.add(line)
    # with open('../coreEntityEmotion_baseline/models/nerDict_pos.txt', 'w', encoding='utf-8') as file:
    #     for entity in entity_set:
    #         file.write('{} {} {}\n'.format(entity, 3, 'n'))
