# coding=utf-8
from pymongo import MongoClient


class DBHelper(object):
    _instance = None
    MONGODB_IP = '127.0.0.1'
    MONGODB_PORT = 27017

    @staticmethod
    def get_instance():
        """
        单例
        :return:
        """
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
        # 反向索引表
        self.reverse_index = self.db.ReverseIndex
        # 关键词表
        self.key_word = self.db.KeyWord

        # 问题反向索引
        self.question_reverse_index=self.db.QuestionReverseIndex

        # 问题正向索引
        self.question_forward_index = self.db.QuestionForwardIndex

        # 获取关键词数目,即最大自增ID
        self.key_word_count = self.key_word.count()

    def get_all_questions(self,mskip=0,mlimit=0):
        """
        查询获得所有问题
        :return:
        """
        return self.questions.find(skip=mskip,limit=mlimit,no_cursor_timeout=True)

    def get_all_answers(self, mskip=0):
        """
        获取所有答案
        :return:
        """
        return self.answers.find(skip=mskip, no_cursor_timeout=True)

    def get_answer_by_id(self, answer_id):
        """
        根据ID获取答案
        :param answer_id: 答案ID
        :return:
        """
        return self.answers.find_one({"answer_id": answer_id})

    def update_answer(self, answer):
        """
        更新答案
        :param answer:答案
        :return:
        """
        self.answers.update({"answer_id": answer['answer_id']},
                            {"$set": {"content": answer['content'], "excerpt": answer['excerpt'],
                                      "len": answer['len']}})

    def update_questions(self, question):
        """
        更新问题信息
        :param question:
        :return:
        """
        self.questions.update({"question_id": question["question_id"]}, {"$set": question})

    def get_forward_indexes(self, mskip=0):
        """
        获得正向索引
        :return:
        """
        return self.forward_index.find(skip=mskip, no_cursor_timeout=True)

    def get_answer_key_word_info_by_id(self,answer_id):
        """
        根据答案ID获取答案的关键词特征(即正向索引)
        :param answer_id:  答案ID
        :return:
        """
        return self.forward_index.find_one({'answer_id':answer_id})['content_info']

    def get_forward_index_count(self):
        """
        获得当前正向索引的数量
        :return:
        """
        return self.forward_index.count()

    def add_forward_indexes(self, answer_id, content_info_dict):
        """
        添加正向索引
        :param answer_id: 答案ID
        :param content_info_dict: 内容信息
        :return: 
        """
        return self.forward_index.insert_one(
            {'answer_id': answer_id, 'content_info': content_info_dict})

    def clear_forward_indexes(self):
        """
        清空正向索引
        :return:
        """
        self.forward_index.remove()

    def clear_reverse_indexes(self):
        """
        清空反向索引
        :return:
        """
        self.reverse_index.remove()

    def clear_key_word(self):
        """
        清空关键词
        :return:
        """
        self.key_word.remove()

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

    def add_reverse_index_content(self, key_word_id, answer_id, weight):
        """
        TODO: 不使用这个结构,可以模仿链表在数据库层面做个链表结构,可能会加快速度
        添加反向索引(内容)
        :param key_word_id:索引关键字ID
        :param answer_id:答案ID
        :param weight:关键字对应权重
        :return:
        """
        self.reverse_index.update_one({"word_key_id": key_word_id}, {"$addToSet": {"content": [answer_id, weight]}},
                                      upsert=True)

    def get_answers_by_key_word_id(self, key_word_id):
        """
        根据关键词ID获取带有关键词的所有答案
        :param key_word_id:关键词ID
        :return:答案ID和对应权重的列表
        """
        res=self.reverse_index.find_one({"word_key_id": key_word_id})
        if res:
            return res['content']
        else:
            return []

    def get_question_by_key_word_id(self,key_word_id):
        """
        根据关键词ID获取带有关键词的所有问题
        :param key_word_id:
        :return:
        """
        res=self.question_reverse_index.find_one({'word_key_id':key_word_id})
        if res:
            return res['content']
        else:
            return []



    def get_key_word_id(self, key_word):
        """
        根据关键词获取关键词ID
        :param key_word: 关键词
        :return:关键词ID,如果查询不到返回None
        """
        r = self.key_word.find_one({"word": key_word})
        if r:
            return r['id']
        else:
            return None

    def get_key_word_by_id(self,key_word_id):
        """
        根据ID返回关键词
        :param key_word_id: 关键词ID
        :return:
        """
        r = self.key_word.find_one({'id':key_word_id})
        if r:
            return r['word']
        else:
            return None

    def add_question_reverse_index_content(self, key_word_id, question_id, weight):
        """
        增加问题反向索引
        :param key_word_id:关键词ID
        :param question_id:问题ID
        :param weight:关键词权重
        :return:
        """
        return self.question_reverse_index.update_one({"word_key_id": key_word_id},{"$addToSet": {"content": [question_id, weight]}},upsert=True)

    def add_question_forward_index_content(self,question_id, content_info_dict):
        """
        增加问题正向索引
        :param question_id:问题ID
        :param content_info_dict:关键词序列
        :return:
        """
        return self.question_forward_index.insert_one({'question_id':question_id,'content_info':content_info_dict})

    def get_question_forward_index(self):
        """
        获取问题正向索引
        :return:
        """
        return self.question_forward_index.find(no_cursor_timeout=True)

    def clear_question_forward_indexes(self):
        """
        清空问题正向索引
        :return:
        """
        self.question_forward_index.remove()


if __name__ == '__main__':
    pass
