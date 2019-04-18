# -*- coding:UTF-8 -*-
"""
@File    : data_split.py
@Time    : 2019/4/15 23:45
@Author  : Blue Keroro
"""
import random


def data_split(path, name):
    content = None
    with open(path + '/' + name, 'r', encoding='utf-8') as f:
        content = f.readlines()
    random.shuffle(content)
    testSize = int(len(content) * 0.2)
    with open(path + '/' + '2_' + name, 'w', encoding='utf-8') as f:
        for line in content[:testSize]:
            f.write(line)
    with open(path + '/' + '8_' + name, 'w', encoding='utf-8') as f:
        for line in content[testSize:]:
            f.write(line)


if __name__ == '__main__':
    data_split('coreEntityEmotion_baseline/data', 'coreEntityEmotion_train.txt')
