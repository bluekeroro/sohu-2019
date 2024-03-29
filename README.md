# [2019 搜狐校园算法大赛](https://biendata.com/competition/sohu2019/)
- 比赛时间： 2019年 4月8日-5月10日 （未进入决赛）
- 排名：48/286  总分数：0.43469 （实体最好得分：0.5508974479041798）
- *该开源部分为核心实体提取部分*
> 竞赛任务  
> 给定若干文章，目标是判断文章的核心实体以及对核心实体的情感态度。
> 每篇文章识别最多三个核心实体，并分别判断文章对上述核心实体的情感倾向（积极、中立、消极三种）。

## 各模块介绍
* bert_bilstm_crf
    * 使用kashgari框架。采用bert预训练模型和bilstm-crf结合的方式。
    * 一般来说，是以输入语料是以句划分。我在这里以文章划分。
    * 标签只有"O", "B-ENT", "I-ENT"，用来标记是否是核心实体。
    * 最终输出每篇文章的核心实体，不限个数。后面将其作为特征之一。
    * 计算输出实体的准确率，即模型输出的正确核心实体占验证集总核心实体的百分比，为41.4105504587156%
* coreEntityEmotion_baseline
    * 官方的baseline
    * 在该模块中处理停用词，nerDict等
    * show_feature.py 分析文本特征
    * coreEntityEmotion_baseline/README.md 中记录特征分析的结果，以及各迭代的准确率
* jieba_fenci_model
    * 结巴预分词模块
    * 先进行分词，以及提取出词的一部分特征，存入文件中，以提高每次运行的效率。
    * 进行多次结巴分词，结合结巴分词的接口tfidf和textRank，提高分词的准确性。具体逻辑可查看jieba_fenci_model/features_ents.py
    * 结巴分词的准确率，定义同上，可达80%以上
* sprint模块为迭代模块，包含算法模型的调整和更新
* sprint_stacking模块（输出了实体0.5508974479041798）
    * 该模块融合lightGBM和xgBoost
    * 每个词的特征选用如下：
        * 在正文中的tfidf分数
        * 在标题中的tfidf分数
        * 在正文中的textRank
        * 在标题中的textRank
        * 实体的长度
        * 词中含有符号的个数
        * 正文中的词频
        * title中的词频
        * 总的词频
        * 关键词第一次出现的位置
        * 关键词最后一次出现的位置
        * 标题的长度
        * 正文的长度
        * 词性
        * 是否在bert_bilstm_crf的输出结果中
        * 语义角色标注的分值
        * 语义角色标注的one-hot矩阵 (*语义角色使用哈工大ltp标注生成，该部分代码不包含在此开源部分中*)
* f1_score.py
    * 按照官网中的分数计算方式，提供计算得分的接口
    
