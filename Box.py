# coding=utf-8

class Box(object):
    """
    打包类,以第一个传入参数为标识符
    """

    def __init__(self, var1, *var2):
        self.var1 = var1
        self.vars = var2

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.var1 == other.var1
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.var1)
