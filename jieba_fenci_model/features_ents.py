import jieba
import jieba.analyse
import re


class feature_ents():
    def __init__(self, ner_dict_path, stopword_file_path):
        self.ner_dict_path = ner_dict_path
        self.stopwords = set()
        with open(stopword_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line.strip())
        jieba.load_userdict(ner_dict_path)
        jieba.analyse.set_stop_words(stopword_file_path)
        self.not_word = '[\n\t，,。`……·\u200b！!?？“”""''~：:;；{}+-——=、/.()（|）%^&*@#$ <>《》【】[]\\]'
        self.key_word_pos = ('n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf',
                             'nt', 'nz', 'nl', 'ng', 'vn', 'v', 'an', 's', 'f', 't', 'tg')
        self.word_pos = dict()

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
        content_words_merge = dict(self.paser_jieba(
            jieba.analyse.extract_tags(content, topK=40, withWeight=True, allowPOS=self.key_word_pos, withFlag=True),
            self.word_pos))  # [(,),...]
        title_words_merge = dict(self.paser_jieba(
            jieba.analyse.extract_tags(title, topK=40, withWeight=True, allowPOS=self.key_word_pos, withFlag=True),
            self.word_pos))
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

    # textRank分数
    def get_textRank_Score(self, news):
        title = news['title']
        content = news['content']
        specialWords = set(re.findall(r'《.*》', title + '\t' + content)) \
                       | set(re.findall(r'[.*]', title + '\t' + content)) \
                       | set(re.findall(r'【.*】', title + '\t' + content))
        for word in specialWords:
            word = word.replace('《', '').replace('》', '').replace('[', '') \
                .replace(']', '').replace('【', '').replace('】', '')
            jieba.add_word(word)
        content_words = self.paser_jieba(jieba.analyse.textrank(content, topK=40, withWeight=True,
                                                                allowPOS=self.key_word_pos, withFlag=True),
                                         self.word_pos)  # [(,),...]
        title_words = self.paser_jieba(
            jieba.analyse.textrank(title, topK=40, withWeight=True, allowPOS=self.key_word_pos, withFlag=True),
            self.word_pos)
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
            self.paser_jieba(
                jieba.analyse.textrank(content, topK=40, withWeight=True, allowPOS=self.key_word_pos, withFlag=True),
                self.word_pos))  # [(,),...]
        title_words_merge = dict(
            self.paser_jieba(
                jieba.analyse.textrank(title, topK=40, withWeight=True, allowPOS=self.key_word_pos, withFlag=True),
                self.word_pos))
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

    def num_of_not_word(self, str_input):
        cnt = 0
        for ch in str_input:
            if ch in self.not_word:
                cnt += 1
        return cnt

    def paser_jieba(self, input, pos_dict):
        ret_list = list()
        for word, score in input:
            ret_list.append((word.word, score))
            pos_dict[word.word] = word.flag
        return ret_list
