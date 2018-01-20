# coding=utf-8
import time
from sklearn import svm
from DBHelper import DBHelper
import jieba
import jieba.analyse
from numpy import *

from Box import Box

from sklearn import preprocessing
import pyltr

# 缩放参数,用于降低问题详细内容中关键词的权重(相对于标题来说)
SHRINK_PARM = 0.5


def learn():
    db = DBHelper.get_instance()

    metric = pyltr.metrics.NDCG(k=10)
    Model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )

    # ----------初始化--------------#


    ALL_X = []
    ALL_Y = []
    ALL_QID=[]

    # ----------初始化--------------#

    questions = db.get_all_questions(1000)  # 训练总问题数

    for question in questions:

        # -----------------提取含有问题关键词的答案列表  TODO:可优化,把关键词直接存到数据库-----------------#
        title_tags = jieba.analyse.extract_tags(question['title'], topK=5, withWeight=True)
        content_tag = jieba.analyse.extract_tags(question['content'], topK=8, withWeight=True)

        tags = title_tags

        # 提取出关键词列表,用于简单去重
        tags_words = [t[0] for t in tags]

        for ct in content_tag:
            if not (ct[0] in tags_words):
                tags.append([ct[0], ct[1] * SHRINK_PARM])

        # 关键词ID列表
        question_key_word_ids = [db.get_key_word_id(t[0]) for t in tags]
        # 关键词权重列表,作为特征值用于模型训练
        question_key_word_weights = [t[1] for t in tags]

        if len(question_key_word_ids) < 8:
            #     特征值不足
            need = 8 - len(question_key_word_ids)
            for i in range(need):
                # 其他补0,凑够8个
                question_key_word_ids.append(0)
                question_key_word_weights.append(0)

        # 包含关键词的所有答案列表(或)

        answers_id_list = []
        for wid in question_key_word_ids:
            o = db.get_answers_by_key_word_id(wid)
            for content in o:
                answers_id_list.append(content[0])

        answers_id_list = set(answers_id_list)

        # -----------------提取含有问题关键词的答案列表-----------------#


        # -----------------计算每个答案的特征值和标准评分-----------------#

        # 数据库读取答案信息(特征值)
        answers_feature = []

        # 标准评分,根据点赞数计算得出
        Y = []

        answers_feature_score_list=[]

        for _id in answers_id_list:
            ans = db.get_answer_by_id(_id)

            # 答案和问题之间的关键词权值的乘积
            key_word_feature = []

            # 数据库获取答案关键词特征值(可能不够15个,需要处理)
            ans_key_word_info = db.get_answer_key_word_info_by_id(_id)

            if ans_key_word_info:
                # 如果关键词特征值不够15个,则后面补0
                if len(ans_key_word_info) < 15:
                    for i in range(15 - len(ans_key_word_info)):
                        ans_key_word_info.append([0, 0])
            if ans_key_word_info:
                # 答案关键词ID列表
                ans_key_word_ids = [info[0] for info in ans_key_word_info]
            else:
                ans_key_word_ids = []

            for i in range(len(ans_key_word_ids)):
                if ans_key_word_ids[i] in question_key_word_ids:
                    # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                    key_word_feature.append(
                        question_key_word_weights[question_key_word_ids.index(ans_key_word_ids[i])] *
                        ans_key_word_info[i][1])
                else:
                    # 如果没有,直接置为0
                    key_word_feature.append(0)
            # 创建时间,更新时间,评论数+关键词特征值(8个)
            # 除了点赞数.点赞数用于评分(输出)

            ft = [ans['create_time'], ans['update_time'], ans['comment_count']]
            ft.extend(key_word_feature)

            temp_y = 0
            for i in key_word_feature:
                temp_y += i * ans['voteup_count']

            answers_feature_score_list.append([_id,temp_y,ft])
            answers_feature.append(ft)



            Y.append(temp_y)
            ALL_X.append(answers_feature)
            ALL_QID.append(question['question_id'])
            ALL_Y.append(Y)

        Model.fit(array(preprocessing.scale(answers_feature)), array(Y))
        print("训练一次")
    # -----------------计算每个答案的特征值和标准评分-----------------#

    questions.close()

    # ---------------测试------------------------#

    new_ques = '游戏行业发展前景如何'

    tags = jieba.analyse.extract_tags(new_ques, topK=8)

    # 问题关键词ID列表
    ques_key_word_ids = [db.get_key_word_id(t[0]) for t in tags]

    ans_id_list = []

    for tag in tags:
        word_id = db.get_key_word_id(tag)
        if word_id:
            # 有关键词
            o = db.get_answers_by_key_word_id(word_id)
            for content in o:
                ans_id_list.append(content[0])

    ans_id_list = set(ans_id_list)

    score_list = []

    for _id in ans_id_list:

        ans = db.get_answer_by_id(_id)

        # 数据库获取答案关键词特征值(可能不够15个,需要处理)
        ans_key_word_info = db.get_answer_key_word_info_by_id(_id)
        # 答案关键词ID列表
        ans_key_word_ids = [_info[0] for _info in ans_key_word_info]

        # 如果关键词特征值不够15个,则后面补0
        if len(ans_key_word_info) < 15:
            for i in range(15 - len(ans_key_word_info)):
                ans_key_word_info.append([0, 0])

        for i in range(len(ans_key_word_ids)):
            if ans_key_word_ids[i] in ques_key_word_ids:
                # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                key_word_feature.append(ques_key_word_ids[i] * ans_key_word_info[i][1])
            else:
                # 如果没有,直接置为0
                key_word_feature.append(0)

        feature = [ans['create_time'], ans['update_time'], ans['comment_count']]
        feature.extend(key_word_feature)

        s = Model.predict(feature)
        score_list.append(Box(_id, s))

    score_list.sort(key=lambda box: box.vars)

    print(score_list)


learn()
