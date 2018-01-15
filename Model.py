# coding=utf-8
import time
from sklearn import svm
from DBHelper import DBHelper
import jieba
import jieba.analyse

# 缩放参数,用于降低问题详细内容中关键词的权重(相对于标题来说)
SHRINK_PARM = 0.5


def learn():
    db = DBHelper.get_instance()
    questions = db.get_all_questions()
    for question in questions:
        title_tags = jieba.analyse.extract_tags(question['title'], topK=5, withWeight=True)
        content_tag = jieba.analyse.extract_tags(question['content'], topK=8, withWeight=True)

        tags = title_tags

        # 提取出关键词列表,用于简单去重
        tags_words = [t[0] for t in tags]

        for ct in content_tag:
            if not (ct[0] in tags_words):
                tags.append([ct[0], ct[1] * SHRINK_PARM])

        # 关键词ID列表
        key_word_ids = [t[0] for t in tags]
        # 关键词权重列表,作为特征值用于模型训练
        weights = [t[1] for t in tags]

        if len(weights) < 8:
            #     特征值不足
            need = 8 - len(weights)
            for i in range(need):
                # 其他补0,凑够8个
                weights.append(0)


        # 包含关键词的所有答案列表(或)

        answers = []
        answers.extend(db.get_answers_by_key_word_id(wid) for wid in key_word_ids)



        X = weights

    questions.close()
