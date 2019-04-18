# -*- coding:UTF-8 -*-
"""
@File    : output.py
@Time    : 2019/4/18 14:43
@Author  : Blue Keroro
"""
from time import time

from sprint3.test import Test

if __name__ == '__main__':
    start = time()
    test = Test()
    test.testCoreEntity('../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1.txt',
                        '../coreEntityEmotion_baseline/data/coreEntityEmotion_test_stage1_result.txt',debug=False)
    print('end:', time() - start)