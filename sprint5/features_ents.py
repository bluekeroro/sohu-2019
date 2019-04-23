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

class feature_ents():
    def __init__(self,ner_dict_path,stopword_file_path):
        self.ner_dict_path = ner_dict_path
        self.stopwords = set()
        self.pku = pkuseg.pkuseg()
        with open(stopword_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line.strip())
        # self.ner = ner()
        jieba.load_userdict(ner_dict_path)
        jieba.analyse.set_stop_words(stopword_file_path)
        self.not_word = '[\n\t，,。`……·\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ <>《》【】[]\\]'
        self.key_word_pos = ('ns', 'n', 'vn', 'v', 'l', 'j', 'nr', 'nrt', 'nt', 'nz', 'nrfg', 'an','s')
        
    # def set_ners(self, news):
    #     self.ners = set(self.ner.pkuseg_cut(news))
        
    # def get_ners(self):
    #     return self.ners
   
    # tfidf分数
    def get_tfidf_Score(self, news):
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
        content_words = jieba.analyse.extract_tags(content, topK=40, withWeight=True)  # [(,),...]
        title_words = jieba.analyse.extract_tags(title, topK=40, withWeight=True)
        content_words_merge = {}
        title_words_merge = {}
        mergeWords = []
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
                mergeWords.append(merge_str)
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
                mergeWords.append(merge_str)
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
        mergeWords = []
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
                mergeWords.append(merge_str)
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
                mergeWords.append(merge_str)
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
                                    len(ner),self.num_of_not_word(ner)]]) # 特征：正文中的tfidf，标题中的tfidf，实体的长度,含有符号的个数
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