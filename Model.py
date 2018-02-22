# coding=utf-8
import time
from sklearn import svm
from DBHelper import DBHelper
import jieba
import jieba.analyse
from numpy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from Box import Box
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from  sklearn.linear_model import ElasticNet

# 缩放参数,用于降低问题详细内容中关键词的权重(相对于标题来说)
SHRINK_PARM = 0.5


def learn():
    db = DBHelper.get_instance()

    X_normalize = preprocessing.Normalizer()
    Y_normalize = preprocessing.MinMaxScaler()

    # Model = MLPRegressor(solver='lbfgs', alpha=1e-40,hidden_layer_sizes=(5, 2), random_state=1)
    Model = ElasticNet(alpha=1e-15, tol=1e-13,max_iter=10000)

    # ----------初始化--------------#


    ALL_X = []
    ALL_Y = []
    ALL_QID = []

    # ----------初始化--------------#

    questions = db.get_all_questions(mlimit=50)  # 训练总问题数

    index = 0

    for question in questions:
        index += 1
        # -----------------提取含有问题关键词的答案列表  TODO:可优化,把关键词直接存到数据库-----------------#
        print(question['title'])
        title_tags = jieba.analyse.extract_tags(question['title'], topK=5, withWeight=True)
        content_tag = jieba.analyse.extract_tags(question['content'], topK=8, withWeight=True)

        title_tags.sort(key=lambda d : d[1])

        tags = title_tags

        # 提取出关键词列表,用于简单去重
        tags_words = [t[0] for t in tags]

        content_tag.sort(key=lambda d:d[1])

        _dta=0
        while len(tags)<8:
            if len(content_tag) > _dta:
                if not (content_tag[_dta][0] in tags_words):
                    tags.append([content_tag[_dta][0],content_tag[_dta][1]*SHRINK_PARM])
                    _dta+=1
                else:
                    _dta+=1
            else:
                tags.append([0,0])


        # 关键词ID列表
        question_key_word_ids = [db.get_key_word_id(t[0]) for t in tags]
        # 关键词权重列表,作为特征值用于模型训练
        question_key_word_weights = [t[1] for t in tags]

        # 权重每个+1,不然会出现小于0的权重,导致相乘的时候分数缩小
        question_key_word_weights = [t + 1 for t in question_key_word_weights]

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


        # 答案ID列表去重
        answers_id_list = set(answers_id_list)


        # -----------------提取含有问题关键词的答案列表-----------------#


        # -----------------提取含有问题关键词的问题列表-----------------#



        # 包含关键词的所有问题列表(或)

        question_word_id_weight_list=[]

        question_id_score_map={}

        for wid in question_key_word_ids:
            o=db.get_question_by_key_word_id(wid)
            for content in o:
                # +1是防止小于1出现相乘越来越小的情况
                _question_id=content[0]
                _weight=content[1]+1

                if _question_id not in question_id_score_map:
                    # 如果问题没有记录,则生成一个,并计算初始分数
                    question_id_score_map[_question_id]=[1,_weight]
                else:
                    # 如果问题之前有记录,计算新的得分
                    _index,_last_weight=question_id_score_map[_question_id]
                    _index+=1
                    _next_weight=(_last_weight+_weight)**_index
                    question_id_score_map[_question_id]=[_index,_next_weight]



        # -----------------提取含有问题关键词的问题列表-----------------#




        # -----------------计算每个答案的特征值和标准评分-----------------#

        # 数据库读取答案信息(特征值)
        answers_feature = []

        # 标准评分,根据点赞数计算得出
        Y = []

        answers_id_score_feature_list = []

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
                ans_key_word_ids = [_info[0] for _info in ans_key_word_info]
            else:
                ans_key_word_ids = []

            for i in range(len(question_key_word_ids)):
                if question_key_word_ids[i] in ans_key_word_ids:
                    # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                    key_word_feature.append(
                        question_key_word_weights[i] *
                        (ans_key_word_info[ans_key_word_ids.index(question_key_word_ids[i])][1]+1))
                else:
                    # 如果没有,直接置为0
                    key_word_feature.append(0)

            # 计算分数

            vp = ans['voteup_count']
            if vp == 0:
                vp = 1
            temp_y = 1
            cy = 0
            for i in key_word_feature:
                if i != 0:
                    temp_y *= i * vp
                    cy += 1

            temp_y = temp_y ** cy


            # 创建时间,更新时间,评论数+关键词特征值(8个)
            # 除了点赞数.点赞数用于评分(输出)

            ft = [ans['create_time'], ans['update_time'], ans['comment_count'],ans['len'],cy]

            ft.extend(key_word_feature)



            answers_id_score_feature_list.append([_id, temp_y, ft])

        # 选出最高分前100
        answers_id_score_feature_list.sort(key=lambda d: d[1], reverse=True)

        each_x=[_i[2] for _i in answers_id_score_feature_list]
        each_y = [_i[1] for _i in answers_id_score_feature_list]

        ll=len(each_x)
        if ll>3:
            ll=3

        print('-------------')
        for _i in range(ll):
            print('id:', answers_id_score_feature_list[_i][0])
            print('score', each_y[_i])
            print('feature', each_x[_i])

        print('-------------')

        each_x=preprocessing.MinMaxScaler().fit_transform(each_x)
        each_y=preprocessing.MinMaxScaler().fit_transform(array(each_y).reshape(-1, 1))



        _len = len(answers_id_score_feature_list)
        if _len > 100:
            _len = 100

        for i in range(_len):
            ALL_X.append(each_x[i])
            ALL_Y.append(each_y[i])
            ALL_QID.append(question['question_id'])

        print('处理问题%d' % (index))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print('开始训练')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    X_normalize = preprocessing.MinMaxScaler().fit(array(ALL_X))
    Y_normalize = preprocessing.MinMaxScaler().fit(array(ALL_Y).reshape(-1, 1))
    XXXXXX = ALL_X
    YYYYYY = ALL_Y
    Model.fit(XXXXXX, YYYYYY)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("训练一次")
    # -----------------计算每个答案的特征值和标准评分-----------------#

    questions.close()

    # ---------------测试------------------------#

    new_ques = '国内有多少人在做量化交易'

    tags = jieba.analyse.extract_tags(new_ques, topK=8, withWeight=True)

    print("测试问题关键词:")
    for t in tags:
        print(t)

    print('\n')

    # 问题关键词ID列表
    ques_key_word_ids = [db.get_key_word_id(t[0]) for t in tags]
    ques_key_word_weights = [t[1] for t in tags]

    ans_id_list = []

    for qid in ques_key_word_ids:
        # 有关键词
        o = db.get_answers_by_key_word_id(qid)
        for content in o:
            ans_id_list.append(content[0])



    # list_all_ans=[]
    # for wid in ques_key_word_ids:
    #     o = db.get_answers_by_key_word_id(wid)
    #     list_all_ans.append([content[0] for content in o])

    # for ans_l in list_all_ans:
    #     if len(list_all_ans)==0:
    #         ans_id_list=ans_l
    #     else:
    #         t_ans_id_list=list(set(ans_l).intersection(set(ans_id_list)))
    #         if len(t_ans_id_list)==0:
    #             break


    for need in range(8-len(ques_key_word_ids)):
        ques_key_word_ids.append(0)
        ques_key_word_weights.append(0)

    ans_id_list = set(ans_id_list)

    score_list = []

    test_ALL_X = []

    for _id in ans_id_list:

        ans = db.get_answer_by_id(_id)
        key_word_feature = []
        feature = []

        # 数据库获取答案关键词特征值(可能不够15个,需要处理)
        ans_key_word_info = db.get_answer_key_word_info_by_id(_id)

        # 如果关键词特征值不够15个,则后面补0
        if len(ans_key_word_info) < 15:
            for i in range(15 - len(ans_key_word_info)):
                ans_key_word_info.append([0, 0])

        # 答案关键词ID列表
        ans_key_word_ids = [_info[0] for _info in ans_key_word_info]

        _i = 0

        for i in range(len(ques_key_word_ids)):
            if ques_key_word_ids[i] in ans_key_word_ids:
                # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                key_word_feature.append(
                    ques_key_word_weights[i] * ((ans_key_word_info[ans_key_word_ids.index(ques_key_word_ids[i])][1])+1))
                _i += 1
            else:
                # 如果没有,直接置为0
                key_word_feature.append(0)

        cy=0
        for _i in key_word_feature:
            if _i!=0:
                cy+=1

        feature = [ans['create_time'], ans['update_time'], ans['comment_count'],ans['len'],cy]
        feature.extend(key_word_feature)

        test_ALL_X.append(feature)


    s = Model.predict(preprocessing.MinMaxScaler().fit_transform(test_ALL_X))
    score_list = np.vstack((array(list(ans_id_list)), array(s)))

    sl = list(score_list.transpose().tolist())

    sl.sort(key=lambda d: d[1], reverse=True)

    for u in sl[0:120]:
        ki = db.get_answer_key_word_info_by_id(u[0])

        print(db.get_answer_by_id(int(u[0]))['voteup_count'])

        for kii in ki:
            print(db.get_key_word_by_id(kii[0]))

        print('内容:\n')
        print(db.get_answer_by_id(int(u[0]))['content'])
        print('\n')


learn()
