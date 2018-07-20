#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import ConfigParser
from tqdm import tqdm


def trainNewModel():
    # print 'Train new model start'
    param = svm_parameter('-s 0 -t 0 -b 1 -q')
    y1, x1 = svm_read_problem('data/traindata_p2p')
    y2, x2 = svm_read_problem('data/traindata_p2G')
    y3, x3 = svm_read_problem('data/traindata_G2G')
    prob1 = svm_problem(y1, x1)
    prob2 = svm_problem(y2, x2)
    prob3 = svm_problem(y3, x3)
    # print '....training p2p'
    model_p2p = svm_train(prob1, param)
    # print '....training p2G'
    model_p2G = svm_train(prob2, param)
    # print '....training G2G'
    model_G2G = svm_train(prob3, param)
    svm_save_model('model/model_R_p2p.model', model_p2p)
    svm_save_model('model/model_R_p2G.model', model_p2G)
    svm_save_model('model/model_R_G2G.model', model_G2G)
    # print 'Train new model finished'

def testModel():
    print('...test model...')
    y1, x1 = svm_read_problem('data/traindata_p2p_origin')
    y2, x2 = svm_read_problem('data/traindata_p2G_origin')
    y3, x3 = svm_read_problem('data/traindata_G2G_origin')
    model_p2p = svm_load_model('model/model_R_p2p.model')
    model_p2G = svm_load_model('model/model_R_p2G.model')
    model_G2G = svm_load_model('model/model_R_G2G.model')
    p1_label, p1_acc, p1_val = svm_predict(y1, x1, model_p2p, '-q')
    acc1, _, _ = evaluations(y1, p1_label)
    p2_label, p2_acc, p2_val = svm_predict(y2, x2, model_p2G, '-q')
    acc2, _, _ = evaluations(y2, p2_label)
    p3_label, p3_acc, p3_val = svm_predict(y3, x3, model_G2G, '-q')
    acc3, _, _ = evaluations(y3, p3_label)
    with open('test.log', 'a') as f:
        print('{}, {}, {}'.format(acc1, acc2, acc3))
        f.write('{}, {}, {}\n'.format(acc1, acc2, acc3))


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')
    a = Dataset.identity_Dataset(config)
    train_album_list_fn = config.get('TEST', 'TRAIN_ALBUM_LIST_FILE')
    a.loadAlbumList(train_album_list_fn)
    data = list(list())
    data.append([0, 0, 0])
    for iteration in tqdm(range(1, 201)):
        # print '====================================================='
        # print 'Iter: %d' % iteration
        if iteration % 10 == 0:
            testModel()
        trainNewModel()
        dataset = a.SimulateDataset(1000, 0.5, 0.4)
        dataset.computeQuality()
        dataset.computeAffinity()
        f = frame.frame()
        f.loadDataset(dataset)
        model_p2p = svm_load_model('model/model_R_p2p.model')
        model_p2G = svm_load_model('model/model_R_p2G.model')
        model_G2G = svm_load_model('model/model_R_G2G.model')
        index = 1
        while f.checkState():
            package = f.getObservation()
            if type(package) == int:
                # print 'Done!'
                break
            data[0] = package
            question_type = len(package)
            if question_type == 3:  #point-----point
                action, t1, t2 = svm_predict([0], data, model_p2p, '-q')
                tp = 'P2P'
            elif question_type == 3 + f.k_size:  #point-----Group or group---point
                action, t1, t2 = svm_predict([0], data, model_p2G, '-q')
                tp = 'P2G'
            else:
                action, t1, t2 = svm_predict([0], data, model_G2G, '-q')
                tp = 'G2G'
            #set action
            if action[0] == 1:
                index += 1
            TF = f.setPerception(action)
            # if TF == False:
            #     print action, index, 1000, f.albumnum, f.queue.qsize(
            #     ), tp, f.dataset.imgID[f.S], f.dataset.imgID[f.D], package
        # f.Normalize_label()
        # f.showResult()
        # raw_input()
    # print 'Done'
