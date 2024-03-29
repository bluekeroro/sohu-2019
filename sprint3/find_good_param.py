# -*- coding:UTF-8 -*-
"""
@File    : find_good_param.py
@Time    : 2019/4/19 0:28
@Author  : Blue Keroro
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tqdm import tqdm

from sprint3.test import Test
from sprint3.validation_accuracy import accuracy


def find_good_param(output_path):
    test = Test()
    with open(output_path, 'w', encoding='utf-8') as file:
        for param in range(10, 100):
            print('param=', param)
            entityScore, emotionScore = accuracy(test, input_param=param / 10)
            file.write(','.join([str(param / 10), str(entityScore), str(emotionScore)]) + '\n')



if __name__ == '__main__':
    from time import time

    start = time()
    find_good_param('param.txt')
    print('end:', time() - start)
