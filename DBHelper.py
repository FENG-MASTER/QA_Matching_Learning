# coding=utf-8
from pymongo import MongoClient


class DBHelper(object):
    _instance = None
    MONGODB_IP = '127.0.0.1'
    MONGODB_PORT = 27017

    @staticmethod
    def get_instance():
        if not DBHelper._instance:
            DBHelper._instance = DBHelper()
        return DBHelper._instance

    def __init__(self):
        self.conn = MongoClient(DBHelper.MONGODB_IP, DBHelper.MONGODB_PORT)
        self.db = self.conn.zhihu3
        self.answers = self.db.AnswerInfo
        self.questions = self.db.QuestionInfo
        # 主题表
        self.topics = self.db.TopicsInfo
        # 正向索引表
        self.forward_index = self.db.ForwardIndex
        # 关键词表
        self.key_word = self.db.KeyWord

        # 获取关键词数目,即最大自增ID
        self.key_word_count = self.key_word.count()

    def get_all_questions(self):
        """
        查询获得所有问题
        :return:
        """
        return self.questions.find(no_cursor_timeout=True)

    def update_questions(self, question):
        """
        更新问题信息
        :param question:
        :return:
        """
        self.questions.update({"question_id": question["question_id"]}, {"$set": question})

    def get_forward_indexes(self):
        """
        获得正向索引
        :return:
        """
        return self.forward_index.find(no_cursor_timeout=True)

    def add_forward_indexes(self, question_id, title_info_dict, content_info_dict):
        """
        添加正向索引
        :param question_id: 问题ID 
        :param title_info_dict: 问题标题信息
        :param content_info_dict: 问题内容信息
        :return: 
        """
        return self.forward_index.insert_one(
            {'question_id': question_id, 'title_info': title_info_dict, 'content_info': content_info_dict})

    def add_many_forward_indexes(self, ques_list):
        """
        添加多个正向索引
        :param ques_list:正向索引列表
        :return:
        """
        self.forward_index.insert_many(
            {'question_id': ques_info[0], 'title_info': ques_info[1], 'content_info': ques_info[2]} for ques_info in
            ques_list)

    def clear_forward_indexes(self):
        """
        清空正向索引
        :return:
        """
        self.forward_index.remove()

    def add_or_get__key_word(self, word):
        """
        增加关键词,如果已经存在,则直接返回关键词ID,如果不存在,新建关键词后返回ID
        :param word:关键词
        :return:
        """
        fres = self.key_word.find_one({"word": word})
        if fres:
            return fres['id']
        else:
            self.key_word_count += 1
            self.key_word.insert_one({"id": self.key_word_count, "word": word})
            return self.key_word_count

    def add_reverse_index_title(self, key_word_id, question_id, weight):
        """
        添加反向索引(标题)
        :param key_word_id:索引关键字ID
        :param question_id:问题ID
        :param weight:关键字对应权重
        :return:
        """


    def add_reverse_index_content(self, key_word_id, question_id, weight):
        """
        添加反向索引(内容)
        :param key_word_id:索引关键字ID
        :param question_id:问题ID
        :param weight:关键字对应权重
        :return:
        """
        pass