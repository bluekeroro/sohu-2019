# -*- coding:UTF-8 -*-
"""
@File    : find_good_param.py
@Time    : 2019/4/19 0:28
@Author  : Blue Keroro
"""
from tqdm import tqdm

from sprint3.test import Test
from sprint3.validation_accuracy import accuracy


def find_good_param(output_path):
    test = Test()
    ret_list = []
    for param in tqdm(range(10, 100)):
        print('param=',param)
        entityScore, emotionScore = accuracy(test, input_param=param / 10)
        ret_list.append([param / 10, entityScore, emotionScore])
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in ret_list:
            f.write(','.join(i) + '\n')


if __name__ == '__main__':
    from time import time

    start = time()
    find_good_param('param.txt')
    print('end:', time() - start)
