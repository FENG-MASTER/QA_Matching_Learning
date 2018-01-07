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
    for question in db_helper.get_all_questions():
        question['content'] = del_html(question['content'])
        db_helper.update_questions(question)


def del_html(content):
    """
    删除html标签,返回删除后文本
    :param content: 原始文本
    :return: 删除标签后文本
    """
    re_html = re.compile('</?\w+[^>]*>')
    s = re_html.sub('', content)
    return s
