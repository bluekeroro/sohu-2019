# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from joblib import load
from sprint1.train import Train
import re
import codecs
from tqdm import tqdm
import re


class Test(Train):
    def __init__(self):
        super(Test, self).__init__()

        self.not_word = '[\n\t，,。`……·\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ 《》【】[]\\]'
        self.param = 3

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
            # content = '昨天下午，redmi正式发布了“小金刚”升级版redminote7pro，这是redmi独立运营后召开的第二场新品发布会，也是卢伟冰入职小米以后的首秀。\n\n首次代表redmi登台亮相的卢伟冰在发布会现场喊出而来“redmi就是性价比之王”的口号，那么redmi的产品表现究竟能不能配得上这个称号呢？在介绍remdinote7pro之前，我们不妨来回顾一下小金刚的表现。小金刚在发布不到一个月便实现了国内销量突破100万的成绩，同时还打破了12个月质保的传统，首次将质保提到了18个月，全新的标准彰显出了小米对品质的信心。同时，小金刚也几乎以一己之力拔高了国内千元机市场对性能、拍照等方面的水准。以小金刚的表现来看，性价比之王的称号已经当之无愧。\n\n不过对于note7的综合表现还是有一些小遗憾，因为这款手机的处理器还是骁龙660，主摄传感器也是用的三星gm1，同时还没有128gb的版本，所以为了填补这份遗憾，redminote7pro应运而生。\n\nredminote7pro除了将处理器升级为对游戏用户更友好的骁龙675之外，最大的亮点就是将相机传感器由三星gm1升级到了索尼imx586。前者是今天发布的vivox27上采用的，999元的小金刚redminote7同款，后者是荣耀v20同款，但是荣耀v20只是单摄，所以成像表现上并不及redminote7pro。而且这两款产品都是3000元起跳的产品。\n\nredminote7pro内置6+128gb的运存组合，电池容量内置4000mah大电池，支持18w快充并标配快充充电头。此外note7pro的整机都加入了p2i生活防泼溅，可以有效防止雨水淋湿、意外溅水等各种问题，降低手机的受潮损坏率。\n\n从上述参数标准可以看出，redminote7pro不仅在其1599元这个价位没有任何对手，而且即便放到友商3000元档位的产品对比中，redminote7pro同样也有自己的优势，所谓的性价比和友商无关，所言非虚。'
            # title = '性价比和友商无关，红米note7 pro诠释什么是性价比之王'
            specialWords = set(re.findall(r'《.*》', title + '\t' + content)) \
                           | set(re.findall(r'[.*]', title + '\t' + content)) \
                           | set(re.findall(r'【.*】', title + '\t' + content))
            for word in specialWords:
                word = word.replace('《', '').replace('》', '').replace('[', '') \
                    .replace(']', '').replace('【', '').replace('】', '')
                jieba.add_word(word)
            content_words = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=())  # [(,),...]
            title_words = jieba.analyse.extract_tags(title, topK=20, withWeight=True, allowPOS=())
            content_words_merge = {}
            title_words_merge = {}
            # ==========sprint1_mergeStr====================
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
                jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=()))  # [(,),...]
            title_words_merge = dict(jieba.analyse.extract_tags(title, topK=20, withWeight=True, allowPOS=()))
            # ==========sprint1_mergeStr====================
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
            content_words_list = sorted(content_words.keys(), key=lambda x: len(x), reverse=True)
            content_words_list = sorted(content_words_list, key=lambda x: content_words[x], reverse=True)
            for word in content_words_list[:3]:
                if (not self.isWord(word)) or ',' in word:
                    raise Exception('非法输出')
            all_entities = [self.delete_mark(word) for word in content_words_list[:3]]
            all_emotions = ['POS' for word in content_words_list[:3]]
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
            str_input = str_input.replace('《','')
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


if __name__ == '__main__':
    test = Test()
    test.testCoreEntity('../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train.txt',
                        '../coreEntityEmotion_baseline/data/2_coreEntityEmotion_train_result.txt')
