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
    * 实体不足两个的数量： 9306
    * 实体长度为一的数量： 9
    * 实体不在nerDict.txt中的数量： 11492
    * 包含空格的实体的数量： 1605
    * 训练集实体总数： 87337
    * 实体不在文章的百分比：0.01030491086252104%
    * 含有非法引号的实体的个数： 7
    * 实体重复率： 55.35798115346302 %
    * 消极实体百分比12.628095766971617%
    * 中立实体百分比37.94153680570663%
    * 积极实体百分比49.430367427321755%
    * coreEntityEmotion_sample_submission_v2.txt可能不是用该baseline计算出来的。该baseline计算出来的实体肯定在nerDict中。
    
* 处理结巴分词：
    * 连续的英文字母和数字不切分
    * nerDict中的专有名词不切分
    * 书名号方括号内不分词
    * 修改结巴分词：使其兼容特殊符号的userDict
    * 修改结巴分词：使jieba.analyse.extract_tags可全模式(不使用全模式)
    * 两次结巴分词merge topK=20 准确率61% ->理想100%
    * 两次结巴分词merge topK=40 准确率70% ->理想100%
    * 两次结巴分词merge topK=60 准确率72.6% ->理想100%
    * 两次结巴分词merge topK=40 增加停用词 准确率70.48% ->理想100%
    * 两次结巴分词merge topK=40 增加停用词 26.38% (textRank allpos)
    * 两次结巴分词merge topK=40 增加停用词 17.41% (textRank)
    * 两次结巴分词merge topK=40 增加停用词 28.56% (tfidf allpos)
    * 两次结巴分词merge topK=40 增加停用词 82.50% (tfidf allpos tfidf_not allpos)
* pkuseg分词测试:
    * 未加stopWord  63.493182886694875%
    * 加stopWord   64.51574988246357%
    * 前40 62.89374706158909%
    * 增加用户词典 43.08885754583921%
    * 去除用户词典 增加news模型 60.836859426422194%
    * 增加用户词典 增加news模型 43.17113305124589%
    * 去除用户词典 增加web模型 64.56276445698167%
    * token_pattern=r"(?u)\b\w+\b" 64.03385049365303%
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
            * 线上：0.260703936019267,0.1282189273105535
    * sprint5:
        * 基于sprint4增加特征：文中的tfidf，标题中的tfidf，实体的长度
            * 线上：0.3984166608185301,0.20206532745479636
            * 线下：0.39561160714285765,0.2049181547619051
            0.384974950396826,0.18681646825396847(过滤非法词)
            0.35...(对特征进行正则化max)
            0.33...(对特征进行正则化l2)
            0.3997,0.2045(不含有符号的个数，jieba增加停用词)
        * 增加特征：文中的tfidf，标题中的tfidf，实体的长度,含有符号的个数(jieba增加停用词)
            * 线下：0.40681250000000013,0.20989136904761943
        * 增加特征：文中的textRank，标题中的textRank，实体的长度,含有符号的个数(jieba增加停用词)
            * 线下：...（bad）
        * 增加特征：文中的tfidf，标题中的tfidf，文中的textRank，标题中的textRank，实体的长度,含有符号的个数(jieba增加停用词)
            * 线下：0.40471428571428625, 0.2119821428571433
            0.43(调参)
    * sprint5_lgb_sklearn:
        * 线下：0.4504776785714291,0.2428764880952385
        
    * sprint6:
        * 使用xgb模型：
            * 线下：0.44224404761904823,0.2323601190476195
    * sprint_stacking:
        * 无参  小数据：
            线下：0.4285602678571434,0.23655803571428563