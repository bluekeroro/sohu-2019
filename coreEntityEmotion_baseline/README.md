# coreEntityEmotion_baseline

> Baseline use entityDict for named entity recognition, you can use a more wise method. Here we use tfIdf score as feature and LR as core Entity classification model <br>
> Baseline use tfIdf vector as emotion features, linearSVC as classfication model <br>

# 总结
> 实体：人、物、地区、机构、团体、企业、行业、某一特定事件等固定存在，且可以作为文章主体的实体词。  
> 核心实体：文章主要描述、或担任文章主要角色的实体词。

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
            * 线上：0.417279706117,0.211698721704
            * 线下：0.6082,0.376  
            0.4453999999999999,0.21859999999999996
        * sprint1_mergeStr: better
            * 线上：0.426700414582,0.217902868056
            * 线下：0.5590,0.311
            0.3998666666666666,0.20113333333333333
    * sprint2:
        * 基于sprint1_mergeStr + cut_all = true + merge (两个实体，如果一个实体是另一个实体的一部分则合并)
            * 线上：0.393441101009,0.20075806548
            * 线下：0.5851323609020904,0.34212522274467505
            0.3803047619047618,0.20363809523809526
    * sprint3:
        * 基于sprint1_mergeStr：循环迭代求参数 best_param = 5.6
            * 线上：0.426757370196,0.217328745895 (param = 5.6)
            0.42914181883527597,0.21849362819736687	(param = 4.9)
            * 线下：0.629026581144777,0.39412771501772614
            0.4198666666666666,0.19853333333333337
    * sprint4:
        * 使用lgb模型：
            * 特征：news正文中的tfidf值，结巴分词再经过pkuseg分词筛选
    * sprint5:
        * 基于sprint4增加特征：文中的tfidf，标题中的tfidf，实体的长度