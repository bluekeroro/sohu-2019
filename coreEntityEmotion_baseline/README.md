# coreEntityEmotion_baseline

> Baseline use entityDict for named entity recognition, you can use a more wise method. Here we use tfIdf score as feature and LR as core Entity classification model <br>
> Baseline use tfIdf vector as emotion features, linearSVC as classfication model <br>

* 训练数据(40000)中文章实体：
    * 没有不包含实体的文章
    * 实体超过三个的数量： 86
    * 实体不足三个的数量： 23476
    * 实体长度为一的数量： 9
    * 实体不在nerDict.txt中的数量： 11492
    * coreEntityEmotion_sample_submission_v2.txt可能不是用该baseline计算出来的。该baseline计算出来的实体肯定在nerDict中。
* 处理结巴分词：
    * 连续的英文字母和数字不切分
    * nerDict中的专有名词不切分
    * 书名号方括号内不分词
    * 修改结巴分词：使其兼容特殊符号的userDict
    * 修改结巴分词：使jieba.analyse.extract_tags可全模式(不使用全模式)
* 处理nerDict:
    * 去除空字符
    * 去除非法词（只包含各种符号的词）
    * 去除词两边的书名号
* 输出实体处理：
    * 去除两边的书名号
    * 不包含非法词
    * 长度小于15
* 本地计算baseline生成数据的F1：
    * turn: 5 
        * 平均entityScore: 0.22016521920111481 
        * 平均emotionScore: 0.0965678793256433
* 迭代:
    * sprint1:
        * sprint1: worst
        * sprint1_mergeStr: better
    * sprint2:
        * 基于sprint1_mergeStr + cut_all = true + merge (两个实体，如果一个实体是另一个实体的一部分则合并)
    