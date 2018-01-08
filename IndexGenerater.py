# coding=utf-8
import time

from DBHelper import DBHelper
import jieba
import jieba.analyse


def generate_forward_index():
    """
    生成问题正向索引,并存入数据库
    索引内容:
        question_id        content_info
        问题ID            内容关键词ID.....

    :return:
    """
    db_helper = DBHelper.get_instance()
    # 清空正向索引
    # db_helper.clear_forward_indexes()

    r = db_helper.get_all_answers(db_helper.get_forward_index_count())
    # 重新生成索引

    i = db_helper.get_forward_index_count()

    # 200W数据跑索引要16小时
    for answer in r:
        res = handler_each_answer(answer)
        db_helper.add_forward_indexes(res[0], res[1])
        i += 1
        print(i)

    r.close()
    # 由于使用行列式生成式的方式,无法关闭游标,所以用以上形式
    # db_helper.add_many_forward_indexes([handler_each_question(question) for question in db_helper.get_all_questions()])


def handler_each_answer(answer):
    """
    处理每个答案,生成正向索引
    :param answer: 答案
    :return: 
    """

    # 答案ID
    id = answer['answer_id']
    # 答案内容文本
    content = answer['content']
    content_info = None
    if len(content) > 1:
        #     有效的内容
        content_info = jieba.analyse.extract_tags(content, topK=8, withWeight=True)
    return id, word_info_to_id_info(content_info)


def word_info_to_id_info(word_info):
    """
    把关键词权重信息转换成ID形式
    :param word_info:关键词信息
    :return:ID形式信息
    """
    if not word_info:
        return None
    db_helper = DBHelper.get_instance()
    return [[db_helper.add_or_get__key_word(w[0]), w[1]] for w in word_info]


def generate_reverse_index():
    """
    生成反向索引(前提是已经生成好了正向索引),索引内容:
    key_word_id                        content
    关键词ID        [[内容包含关键词的答案1的ID,关键词权重],........]

    :return:
    """
    db_helper = DBHelper.get_instance()
    for f_index in db_helper.get_forward_indexes():
        a_id = f_index['answer_id']
        content_info = f_index['content_info']
        if content_info:
            for content in content_info:
                db_helper.add_reverse_index_content(content[0], a_id, content[1])


generate_forward_index()
print("正向结束")
generate_reverse_index()
