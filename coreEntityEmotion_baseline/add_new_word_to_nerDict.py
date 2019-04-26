# -*- coding:UTF-8 -*-
"""
@File    : add_new_word_to_nerDict.py
@Time    : 2019/4/25 22:01
@Author  : Blue Keroro
"""
import os

from tqdm import tqdm

if __name__ == '__main__':
    ner = set()
    with open('models/nerDict.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            ner.add(line)
    filePath = 'C:/Users/陈伟/PycharmProjects/sohu-2019/coreEntityEmotion_baseline/models/new_words'
    txt_list = os.listdir(filePath)
    print('读取新词文件...')
    for txt in tqdm(txt_list):
        with open(filePath + '/' + txt, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                ner.add(line)
    print('写入新词...')
    with open('models/nerDict.txt', 'w', encoding='utf-8') as file:
        for word in tqdm(ner):
            file.write(word + '\n')
