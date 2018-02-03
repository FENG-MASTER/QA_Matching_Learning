# coding=utf-8
import pyltr

from  sklearn.linear_model import ElasticNet

Model = ElasticNet(alpha=1e-40, tol=1e-38, max_iter=10000)

Model.fit([[1,1,6],[4,4,5],[4,5,8],[3,4,7]],[1,3,3,4])

print(Model.predict([[3,4,3],[2,3,6]]))
