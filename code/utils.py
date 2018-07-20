from sys import path
path.append('../libsvm/python')
from svmutil import *

def init_xgboost_model():
    y1, x1 = svm_read_problem('../data/traindata_p2p_origin')
    y2, x2 = svm_read_problem('../data/traindata_p2G_origin')
    y3, x3 = svm_read_problem('../data/traindata_G2G_origin')
    model_p2p = svm_load_model('../model/model_R_p2p.model')
    model_p2G = svm_load_model('../model/model_R_p2G.model')
    model_G2G = svm_load_model('../model/model_R_G2G.model')
    p1_label, p1_acc, p1_val = svm_predict(y1, x1, model_p2p, '-q -b 1')
    p2_label, p2_acc, p2_val = svm_predict(y2, x2, model_p2G, '-q -b 1')
    p3_label, p3_acc, p3_val = svm_predict(y3, x3, model_G2G, '-q -b 1')
    print(p1_val[:10])

if __name__ == '__main__':
    init_xgboost_model()