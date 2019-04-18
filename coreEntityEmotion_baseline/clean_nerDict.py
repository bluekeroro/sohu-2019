# -*- coding:UTF-8 -*-
"""
@File    : clean_nerDict.py
@Time    : 2019/4/18 18:59
@Author  : Blue Keroro
"""

not_word = '[\n\t，,。\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ 《》【】[]\\]'


def clean(nerDict_path, output_path):
    # nerDict_path = 'models/nerDict.txt'
    nerCorpus = []
    with open(nerDict_path, 'r', encoding='utf-8') as f:
        for word in f:
            word = word.strip()
            nerCorpus.append(word)
    for index, word in enumerate(nerCorpus):
        if len(word) == 0 or word[0] in not_word or word[-1] in not_word:
            print(index, word)
    delete_index_set = set()
    for index, word in enumerate(nerCorpus):
        if len(word) == 0:
            delete_index_set.add(index)
            continue
        if not isWord(word):
            delete_index_set.add(index)
        if word[0] == '《' and word[-1] == '》':
            nerCorpus[index] = word[1:-1]
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, word in enumerate(nerCorpus):
            if index in delete_index_set:
                continue
            f.write(word.replace('}','').replace('{','').replace('/','').replace('$','').replace('^','')+'\n')

def isWord(str_input):
    for char in str_input:
        if char not in not_word:
            return True
    return False


if __name__ == '__main__':
    clean('models/nerDict00.txt', 'models/nerDict.txt')
