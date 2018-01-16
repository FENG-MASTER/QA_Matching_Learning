# coding=utf-8
import time
from sklearn import svm
from DBHelper import DBHelper
import jieba
import jieba.analyse

# 缩放参数,用于降低问题详细内容中关键词的权重(相对于标题来说)
SHRINK_PARM = 0.5

# 等待计算的权重(1开始)
weight = []

index = []
tao = []
learningRate = 0.0003
# 特征值纬度
weidu = 6
# 迭代次数
iter_num = 30


def learn():
    db = DBHelper.get_instance()

    # ----------初始化--------------#

    # 初始化权重
    weight = [1.0 for n in range(weidu + 2)]

    # ----------初始化--------------#

    questions = db.get_all_questions()

    # 迭代次数
    for iter in range(iter_num):

        # 训练总问题数
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
            question_key_word_ids = [t[0] for t in tags]
            # 关键词权重列表,作为特征值用于模型训练
            question_key_word_weights = [t[1] for t in tags]

            if len(question_key_word_weights) < 8:
                #     特征值不足
                need = 8 - len(question_key_word_weights)
                for i in range(need):
                    # 其他补0,凑够8个
                    question_key_word_weights.append(0)

            # 包含关键词的所有答案列表(或)

            answers_ids = []
            answers_ids.extend(db.get_answers_by_key_word_id(wid) for wid in question_key_word_ids)

            # -----------------提取含有问题关键词的答案列表-----------------#


            # -----------------计算每个答案的特征值-----------------#

            # 数据库读取答案信息(特征值)
            answers_feature = []

            # 答案和问题之间的关键词权值的乘积
            key_word_feature = []

            for id in answers_ids:
                ans = db.get_answer_by_id(id)

                # 数据库获取答案关键词特征值(可能不够15个,需要处理)
                ans_key_word_info = db.get_answer_key_word_info_by_id(id)
                # 答案关键词ID列表
                ans_key_word_ids = [info[0] for info in ans_key_word_info]

                # 如果关键词特征值不够15个,则后面补0
                if len(ans_key_word_info) < 15:
                    for i in range(15 - len(ans_key_word_info)):
                        ans_key_word_info.append([0, 0])

                for i in range(len(ans_key_word_ids)):
                    if ans_key_word_ids[i] in question_key_word_ids:
                        # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                        key_word_feature.append(question_key_word_weights[i] * ans_key_word_info[i][1])
                    else:
                        # 如果没有,直接置为0
                        key_word_feature.append(0)
                # 创建时间,更新时间,点赞数,评论数+关键词特征值(8个)
                answers_feature.append(
                    [ans['create_time'], ans['update_time'], ans['voteup_count'], ans['comment_count']].extend(
                        key_word_feature))

            # -----------------计算每个答案的特征值-----------------#


            # 权重修正值
            delta_w = [0 for i in range(weidu + 2)]

            # 答案列表个数
            len_of_answers = len(answers_ids)

            # 答案评分数组,每个答案对应一个评分
            # 初始化
            fw = [0 for i in range(len_of_answers)]

            # 第一步 算得分数组fw
            for k in range(1, len_of_answers):
                for p in range(1, weidu):
                    fw[k] = fw[k] + weight[p]

            X = question_key_word_weights

    questions.close()
