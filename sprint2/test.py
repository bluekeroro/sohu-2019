# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from joblib import load
from sprint2.train import Train
import re
import codecs
from tqdm import tqdm
import re


class Test(Train):
    def __init__(self):
        super(Test, self).__init__()

        self.not_word = '[\n\t，,。`……·\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ 《》【】[]\\]'
        self.param = 5.6

    def testCoreEntity(self, test_path, result_path, param=None, debug=True):
        if param != None:
            self.param = param
        testData = self.loadData(test_path)
        if debug:
            testData = testData[:100]

        f_submit = codecs.open(result_path,
                               'w', 'utf-8')

        for news in tqdm(testData):
            title = news['title']
            content = news['content']
            # content = "3月25日消息，据国外媒体报道，在上周上调models、model3等多款电动汽车的售价之后，电动汽车厂商特斯拉目前也对尚未开售的modely的售价进行了调整，目前接受预订的三个版本的起售价均上调了1000美元。\n\nmodely是特斯拉在3月14日推出的跨界suv，共有标准续航版、后轮驱动长续航版、双电机全轮驱动板长续航版和双电机全轮驱动高性能版4个版本，标准续航版预计在2021年量产，另外三个版本预计在2020年生产。在3月14日发布时，特斯拉公布的modely的起售价分别是3.9万美元、4.7万美元、5.1万美元和6万美元。\n据知名财经媒体《财富》杂志报道，加拿大皇家银行（rbc）分析师约瑟夫·斯帕克（josephspak）周一在给投资者的一份报告中表示，他现在预计特斯拉今年的model3汽车发货量将达到26.1万辆，低于cnbc此前估计的26.8万辆。\nspak在致投资者的报告中表示，特斯拉第一季度在中国处理了一个“海关问题”，可能会影响其出货量。他还将第一季度model3的出货量预测从全球57000台下调至52500台。最终，spak将特斯拉股票的12个月目标价从245美元下调至210美元。\n感谢您的查看，你们买电动汽车会买特斯拉吗？你们有什么想说的吗？"
            # title = "特斯拉上调model y售价，特斯拉还ok吗？"
            specialWords = set(re.findall(r'《.*》', title + '\t' + content)) \
                           | set(re.findall(r'[.*]', title + '\t' + content)) \
                           | set(re.findall(r'【.*】', title + '\t' + content))
            for word in specialWords:
                word = word.replace('《', '').replace('》', '').replace('[', '') \
                    .replace(']', '').replace('【', '').replace('】', '')
                jieba.add_word(word)
            content_words = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=(),
                                                       cut_all=True)  # [(,),...]
            title_words = jieba.analyse.extract_tags(title, topK=20, withWeight=True, allowPOS=(), cut_all=True)
            content_words_merge = {}
            title_words_merge = {}
            mergeWords = []
            for index, word in enumerate(content_words):
                if index > 0 and word[1] == content_words[index - 1][1]:
                    if (not self.isWord(word[0])) or (not self.isWord(content_words[index - 1][0])):
                        continue
                    merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(content_words[index - 1][0])
                    tmp_index = index + 1
                    while tmp_index < len(content_words) and content_words[tmp_index][1] == word[1]:
                        if not self.isWord(content_words[tmp_index][0]):
                            break
                        merge_str += '.*' + self.ban_escape_char(content_words[tmp_index][0])
                        tmp_index += 1
                    mergeWords.append(merge_str)
            for index, word in enumerate(title_words):
                if index > 0 and word[1] == title_words[index - 1][1]:
                    if (not self.isWord(word[0])) or (not self.isWord(title_words[index - 1][0])):
                        continue
                    merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(title_words[index - 1][0])
                    tmp_index = index + 1
                    while tmp_index < len(title_words) and title_words[tmp_index][1] == word[1]:
                        if not self.isWord(title_words[tmp_index][0]):
                            break
                        merge_str += '.*' + self.ban_escape_char(title_words[tmp_index][0])
                        tmp_index += 1
                    mergeWords.append(merge_str)
            for mergeWord in mergeWords:
                re_words = re.findall(mergeWord, title + '\t' + content)
                for re_word in re_words:
                    if len(re_word) > 0:
                        jieba.add_word(re_word)
            content_words_merge = dict(
                jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=(), cut_all=True))  # [(,),...]
            title_words_merge = dict(
                jieba.analyse.extract_tags(title, topK=20, withWeight=True, allowPOS=(), cut_all=True))
            content_words = dict(content_words)
            content_words.update(content_words_merge)
            title_words = dict(title_words)
            title_words.update(title_words_merge)
            for word in title_words:
                if word in content_words:
                    content_words[word] += title_words[word] / self.param
                else:
                    content_words[word] = title_words[word] / self.param
            for word in content_words:
                if (not self.isWord(word)) or ',' in word or len(word) > 15:  # 不输出带英文逗号的实体 限制实体的长度
                    content_words[word] = 0
            self.merge_word(content_words)
            content_words_list = sorted(content_words.keys(), key=lambda x: len(x), reverse=True)
            content_words_list = sorted(content_words_list, key=lambda x: content_words[x], reverse=True)
            for word in content_words_list[:3]:
                if (not self.isWord(word)) or ',' in word:
                    raise Exception('非法输出')
            all_entities = [self.delete_mark(word) for word in content_words_list[:3] if len(word) > 0]
            all_emotions = ['POS' for entity in all_entities]
            f_submit.write(news['newsId'])
            f_submit.write('\t')
            f_submit.write(','.join(all_entities))
            f_submit.write('\t')
            f_submit.write(','.join(all_emotions))
            f_submit.write('\n')

    def isWord(self, str_input):
        for char in str_input:
            if char not in self.not_word:
                return True
        return False

    def ban_escape_char(self, str_input):
        str_list = list(str_input)
        ret_str = ''
        for ch in str_list:
            if ch in self.not_word:
                ret_str += '\\'
            ret_str += ch
        return ret_str

    def delete_mark(self, str_input):
        # 处理书名号
        if str_input[0] == '《' and str_input[-1] == '》':
            return str_input[1:-1]
        if '《' in str_input and '》' not in str_input:
            str_input = str_input.replace('《', '')
        elif '》' in str_input and '《' not in str_input:
            str_input = str_input.replace('》', '')
        str_input = str_input.replace('\'', '')
        str_input = str_input.replace('\"', '')  # 如果实体中只含有一个引号，会导致提交报错
        return str_input
        # 不能直接删除左右两端的符号
        # left = 0
        # right = len(str_input)
        # while left < len(str_input) and str_input[left] in self.not_word:
        #     left += 1
        # while right > 0 and str_input[right - 1] in self.not_word:
        #     right -= 1
        # if left >=right:
        #     raise Exception('该单词全是不合法字符')
        # return str_input[left:right]

    def merge_word(self, content_words):
        keys_list = content_words.keys()
        for word in content_words:
            if content_words[word] == 0:
                continue
            for key in keys_list:
                if word in key and len(word) < len(key) and content_words[key] != 0:
                    content_words[key] = max(content_words[key], content_words[word])
                    content_words[word] = 0


if __name__ == '__main__':
    test = Test()
    test.testCoreEntity('../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train.txt',
                        '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
