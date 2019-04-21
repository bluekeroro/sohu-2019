import jieba
import jieba.analyse
import pkuseg
from joblib import dump,load
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

        
    # def set_ners(self, news):
    #     self.ners = set(self.ner.pkuseg_cut(news))
        
    # def get_ners(self):
    #     return self.ners
   
    # tfidf分数
    def get_tfidf_Score(self, news):
        title = news['title']
        content = news['content']
        article = title+ ' '+content
        tfidf = {}
        for ner, score in jieba.analyse.extract_tags(content, topK=50, withWeight=True):
            tfidf[ner] = score
        return tfidf
    
    
    # 把特征接到一起
    def combine_features(self, news):
        tfidf = self.get_tfidf_Score(news)
        
        features = []
        for ner in self.pkuseg_cut(news):
            a = 0
            if ner in tfidf: #0
                a = tfidf[ner] 
            features.append([[ner],[a]])  #特征可以继续添加 b,c,d,e,f,g......
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
