import jieba
import jieba.analyse
import pkuseg
from joblib import dump,load
from sklearn.preprocessing import normalize
from tqdm import tqdm
import time
import json
import re
import numpy as np

from jieba_fenci_model import fenci
import jieba.posseg as pseg

class feature_ents():
    def __init__(self,ner_dict_path,stopword_file_path,load_from_file=False):
        self.ner_dict_path = ner_dict_path
        self.stopwords = set()
        self.pku = pkuseg.pkuseg()
        with open(stopword_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line.strip())
        # self.ner = ner()
        self.load_from_file = load_from_file
        jieba.load_userdict(ner_dict_path)
        jieba.analyse.set_stop_words(stopword_file_path)
        self.not_word = '[\n\t，,。`……·\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ <>《》【】[]\\]'
        self.key_word_pos = ('ns', 'n', 'vn', 'v', 'nr', 'nt', 'nz', 'an','s')
        self.feature_data_dict = None
        
    # def set_ners(self, news):
    #     self.ners = set(self.ner.pkuseg_cut(news))
        
    # def get_ners(self):
    #     return self.ners
   
    # tfidf分数
    def get_tfidf_Score(self, news):
        title = news['title']
        content = news['content']
        specialWords = set(re.findall(r'《.*》', title + '\t' + content)) \
                       | set(re.findall(r'[.*]', title + '\t' + content)) \
                       | set(re.findall(r'【.*】', title + '\t' + content))
        for word in specialWords:
            word = word.replace('《', '').replace('》', '').replace('[', '') \
                .replace(']', '').replace('【', '').replace('】', '')
            jieba.add_word(word)
        content_words = jieba.analyse.extract_tags(content, topK=40, withWeight=True)  # [(,),...]
        title_words = jieba.analyse.extract_tags(title, topK=40, withWeight=True)
        content_words_merge = {}
        title_words_merge = {}
        mergeWords = set()
        for index, word in enumerate(content_words):
            if index > 0 and word[1] == content_words[index - 1][1] and word[1]>0:
                if (not self.isWord(word[0])) or (not self.isWord(content_words[index - 1][0])):
                    continue
                merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(content_words[index - 1][0])
                tmp_index = index + 1
                while tmp_index < len(content_words) and content_words[tmp_index][1] == word[1]:
                    if not self.isWord(content_words[tmp_index][0]):
                        break
                    merge_str += '.*' + self.ban_escape_char(content_words[tmp_index][0])
                    tmp_index += 1
                mergeWords.add(merge_str)
        for index, word in enumerate(title_words):
            if index > 0 and word[1] == title_words[index - 1][1] and word[1]>0:
                if (not self.isWord(word[0])) or (not self.isWord(title_words[index - 1][0])):
                    continue
                merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(title_words[index - 1][0])
                tmp_index = index + 1
                while tmp_index < len(title_words) and title_words[tmp_index][1] == word[1]:
                    if not self.isWord(title_words[tmp_index][0]):
                        break
                    merge_str += '.*' + self.ban_escape_char(title_words[tmp_index][0])
                    tmp_index += 1
                mergeWords.add(merge_str)
        for mergeWord in mergeWords:
            re_words = re.findall(mergeWord, title + '\t' + content)
            for re_word in re_words:
                if len(re_word) > 0:
                    jieba.add_word(re_word)
        content_words_merge = dict(
            jieba.analyse.extract_tags(content, topK=40, withWeight=True))  # [(,),...]
        title_words_merge = dict(jieba.analyse.extract_tags(title, topK=40, withWeight=True))
        content_words = dict(content_words)
        content_words.update(content_words_merge)
        title_words = dict(title_words)
        title_words.update(title_words_merge)
        for key in list(content_words):
            if not self.isWord(key):
                content_words.pop(key)
        for key in list(title_words):
            if not self.isWord(key):
                title_words.pop(key)
        return content_words,title_words

    # textRank分数
    def get_textRank_Score(self,news):
        title = news['title']
        content = news['content']
        specialWords = set(re.findall(r'《.*》', title + '\t' + content)) \
                       | set(re.findall(r'[.*]', title + '\t' + content)) \
                       | set(re.findall(r'【.*】', title + '\t' + content))
        for word in specialWords:
            word = word.replace('《', '').replace('》', '').replace('[', '') \
                .replace(']', '').replace('【', '').replace('】', '')
            jieba.add_word(word)
        content_words = jieba.analyse.textrank(content, topK=40, withWeight=True,allowPOS=self.key_word_pos)  # [(,),...]
        title_words = jieba.analyse.textrank(title, topK=40, withWeight=True,allowPOS=self.key_word_pos)
        content_words_merge = {}
        title_words_merge = {}
        mergeWords = set()
        for index, word in enumerate(content_words):
            if index > 0 and word[1] == content_words[index - 1][1] and word[1] > 0:
                if (not self.isWord(word[0])) or (not self.isWord(content_words[index - 1][0])):
                    continue
                merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(content_words[index - 1][0])
                tmp_index = index + 1
                while tmp_index < len(content_words) and content_words[tmp_index][1] == word[1]:
                    if not self.isWord(content_words[tmp_index][0]):
                        break
                    merge_str += '.*' + self.ban_escape_char(content_words[tmp_index][0])
                    tmp_index += 1
                mergeWords.add(merge_str)
        for index, word in enumerate(title_words):
            if index > 0 and word[1] == title_words[index - 1][1] and word[1] > 0:
                if (not self.isWord(word[0])) or (not self.isWord(title_words[index - 1][0])):
                    continue
                merge_str = self.ban_escape_char(word[0]) + '.*' + self.ban_escape_char(title_words[index - 1][0])
                tmp_index = index + 1
                while tmp_index < len(title_words) and title_words[tmp_index][1] == word[1]:
                    if not self.isWord(title_words[tmp_index][0]):
                        break
                    merge_str += '.*' + self.ban_escape_char(title_words[tmp_index][0])
                    tmp_index += 1
                mergeWords.add(merge_str)
        for mergeWord in mergeWords:
            re_words = re.findall(mergeWord, title + '\t' + content)
            for re_word in re_words:
                if len(re_word) > 0:
                    jieba.add_word(re_word)
        content_words_merge = dict(
            jieba.analyse.textrank(content, topK=40, withWeight=True,allowPOS=self.key_word_pos))  # [(,),...]
        title_words_merge = dict(jieba.analyse.textrank(title, topK=40, withWeight=True,allowPOS=self.key_word_pos))
        content_words = dict(content_words)
        content_words.update(content_words_merge)
        title_words = dict(title_words)
        title_words.update(title_words_merge)
        for key in list(content_words):
            if not self.isWord(key):
                content_words.pop(key)
        for key in list(title_words):
            if not self.isWord(key):
                title_words.pop(key)
        return content_words, title_words
    
    # 把特征接到一起
    def combine_features(self, news):
        features = []
        if self.load_from_file is True:
            if self.feature_data_dict is None:
                self.feature_data_dict = fenci.get_fenci_feature_func(
                    '../jieba_fenci_model/result/result_jieba_fenci.txt')
            for ner in self.feature_data_dict[news['newsId']]:
                features.append(
                    [[ner], self.feature_data_dict[news['newsId']][ner] + [len(ner),
                                                                           self.num_of_not_word(ner),
                                                                           news['content'].count(ner),  # 正文中的词频
                                                                           news['title'].count(ner),  # title中的词频
                                                                           (news['title'] + news['content']).count(ner),
                                                                           # 总的词频
                                                                           (news['title'] + news['content']).index(ner),
                                                                           # 关键词第一次出现的位置
                                                                           (news['title'] + news['content']).rindex(
                                                                               ner),  # 关键词最后一次出现的位置
                                                                           len(news['title']),  # 标题的长度
                                                                           len(news['content'])  # 正文的长度
                                                                           ]])
            return features
        content_words_tfidf, title_words_tfidf = self.get_tfidf_Score(news)
        content_words_textRank, title_words_textRank = self.get_textRank_Score(news)
        # content_words_tfidf = content_words_textRank
        # title_words_tfidf = title_words_textRank
        keys = content_words_tfidf.keys()|title_words_tfidf.keys()|content_words_textRank.keys()|title_words_textRank.keys()
        features = []
        for ner in keys:
            features.append([[ner],[content_words_tfidf[ner] if ner in content_words_tfidf else 0,
                                    title_words_tfidf[ner] if ner in title_words_tfidf else 0,
                                    content_words_textRank[ner] if ner in content_words_textRank else 0,
                                    title_words_textRank[ner] if ner in title_words_textRank else 0,
                                    len(ner),self.num_of_not_word(ner),  # 特征：正文中的tfidf，标题中的tfidf，实体的长度,含有符号的个数
                                    news['content'].count(ner),  # 正文中的词频
                                    news['title'].count(ner),  # title中的词频
                                    (news['title'] + news['content']).count(ner),  # 总的词频
                                    (news['title'] + news['content']).index(ner),  # 关键词第一次出现的位置
                                    (news['title'] + news['content']).rindex(ner),  # 关键词最后一次出现的位置
                                    len(news['title']),  # 标题的长度
                                    len(news['content'])  # 正文的长度
                                    ]])
        # self.num_of_not_word(ner)
        # 正则化 （效果差）
        # feature_matrix = [feature[1] for feature in features]
        # feature_matrix = normalize(np.array(feature_matrix))
        # for index, feature in enumerate(features):
        #     feature[1] = list(feature_matrix[index])

        # for ner in self.pkuseg_cut(news):
        #     a = 0
        #     if ner in tfidf: #0
        #         a = tfidf[ner]
        #     features.append([[ner],[a]])  #特征可以继续添加 b,c,d,e,f,g......
        return features

    def pkuseg_cut(self, news):
        title = news['title']
        content = news['content']
        nerdict = set()

        sentences = []
        for seq in re.split(r'[\n。，、：‘’“""”？！?!《》]', title + " " + content):
            seq = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", seq)
            if len(seq) > 2:
                sentences.append(seq)
        for seq in sentences:
            ner_list = self.pku.cut(seq)
            for ner in ner_list:
                if len(ner) > 1:
                    if ner not in self.stopwords:  # 去停用词
                        nerdict.add(ner)
        return nerdict  # set

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

    def num_of_not_word(self,str_input):
        cnt=0
        for ch in str_input:
            if ch in self.not_word:
                cnt+=1
        return cnt