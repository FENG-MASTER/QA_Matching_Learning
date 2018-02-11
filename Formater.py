# coding=utf-8
# 用于格式化抓取的数据
import re
from DBHelper import DBHelper


def del_question_content_html_from_db():
    """
    删除数据库中所有问题内容中的html标签
    :return:
    """
    db_helper = DBHelper.get_instance()
    r = db_helper.get_all_questions()
    for question in r:
        question['content'] = del_html(question['content'])
        db_helper.update_questions(question)
    r.close()


def del_html(content):
    """
    删除html标签,返回删除后文本
    :param content: 原始文本
    :return: 删除标签后文本
    """
    re_html = re.compile('</?\w+[^>]*>')
    s = re_html.sub('', content)
    return s


def del_answer_content_html_from_db():
    """
    删除数据库中所有答案内容中的html标签
    :return:
    """
    i = 206000
    db_helper = DBHelper.get_instance()
    r = db_helper.get_all_answers(206000)
    for answer in r:
        answer['content'] = del_html(answer['content'])
        answer['excerpt'] = del_html(answer['excerpt'])
        db_helper.update_answer(answer)
        i += 1
        if i % 1000 == 0:
            print(i)
    r.close()


def cal_all_answer_len():
    """
    计算出每个答案的文本长度并写入数据库
    :return:
    """
    db_helper = DBHelper.get_instance()
    r = db_helper.get_all_answers()
    i=0
    for answer in r:
        i+=1
        answer['len'] = len(answer['content'])
        db_helper.update_answer(answer)
        if i%1000==0:
            print(i)
    r.close()


def add_question_key_word_info():
    """
    添加问题关键词信息到问题数据库,防止多次计算,方便使用
    :return: 
    """
    db_helper=DBHelper.get_instance()
    r=db_helper.get_all_questions()
    i=0
    for question in r:
        i+=1


cal_all_answer_len()
