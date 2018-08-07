#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import ConfigParser
from tqdm import tqdm
import os
import shutil
import time
from replay_buffer import ReplayBufferSimple
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


def get_data(fn):
    data = load_svmlight_file(fn)
    return data[0], data[1]


def train_ssvm_model(config, f, models):
    p2p_traindata = config.get('REID', 'REWARD_P2P_MODEL_TRAIN_DATA')
    p2g_traindata = config.get('REID', 'REWARD_P2G_MODEL_TRAIN_DATA')
    g2g_traindata = config.get('REID', 'REWARD_G2G_MODEL_TRAIN_DATA')
    train_data = [p2p_traindata, p2g_traindata, g2g_traindata]
    for model, rb, fn in zip(models, f.replay_buffers, train_data):
        rb.sample2(fn, 100)
        X, y = get_data(fn)
        model.partial_fit(X, y, classes=[0, 1])


def test_ssvm_model(config, models):
    p2p_testdata = config.get('REID',
                              'REWARD_P2P_MODEL_TRAIN_DATA') + '_origin'
    p2g_testdata = config.get('REID',
                              'REWARD_P2G_MODEL_TRAIN_DATA') + '_origin'
    g2g_testdata = config.get('REID',
                              'REWARD_G2G_MODEL_TRAIN_DATA') + '_origin'
    if not os.path.exists(p2p_testdata):
        print('...test files not exist, create by copy...')
        shutil.copyfile(p2p_testdata[:-7], p2p_testdata)
        shutil.copyfile(p2g_testdata[:-7], p2g_testdata)
        shutil.copyfile(g2g_testdata[:-7], g2g_testdata)

    test_data = [p2p_testdata, p2g_testdata, g2g_testdata]

    accs = []
    for model, fn in zip(models, test_data):
        X, y = get_data(fn)
        accs.append(model.score(X, y))

    print(accs)


def trainNewModel(config):
    # print 'Train new model start'
    param = svm_parameter('-s 0 -t 0 -b 1 -q')
    y1, x1 = svm_read_problem(
        config.get('REID', 'REWARD_P2P_MODEL_TRAIN_DATA'))
    y2, x2 = svm_read_problem(
        config.get('REID', 'REWARD_P2G_MODEL_TRAIN_DATA'))
    y3, x3 = svm_read_problem(
        config.get('REID', 'REWARD_G2G_MODEL_TRAIN_DATA'))
    prob1 = svm_problem(y1, x1)
    prob2 = svm_problem(y2, x2)
    prob3 = svm_problem(y3, x3)
    # print '....training p2p'
    model_p2p = svm_train(prob1, param)
    # print '....training p2G'
    model_p2g = svm_train(prob2, param)
    # print '....training G2G'
    model_g2g = svm_train(prob3, param)
    svm_save_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_p2p.model'), model_p2p)
    svm_save_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_p2g.model'), model_p2g)
    svm_save_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_g2g.model'), model_g2g)
    # print 'Train new model finished'


def testModel(config):
    print('...test model...')
    test_r_p2p_data = config.get('REID',
                                 'REWARD_P2P_MODEL_TRAIN_DATA') + '_origin'
    test_r_p2g_data = config.get('REID',
                                 'REWARD_P2G_MODEL_TRAIN_DATA') + '_origin'
    test_r_g2g_data = config.get('REID',
                                 'REWARD_G2G_MODEL_TRAIN_DATA') + '_origin'
    if not os.path.exists(test_r_p2p_data):
        print('...test files not exist, create by copy...')
        shutil.copyfile(test_r_p2p_data[:-7], test_r_p2p_data)
        shutil.copyfile(test_r_p2g_data[:-7], test_r_p2g_data)
        shutil.copyfile(test_r_g2g_data[:-7], test_r_g2g_data)
    y1, x1 = svm_read_problem(test_r_p2p_data)
    y2, x2 = svm_read_problem(test_r_p2g_data)
    y3, x3 = svm_read_problem(test_r_g2g_data)
    model_p2p = svm_load_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_p2p.model'))
    model_p2G = svm_load_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_p2g.model'))
    model_G2G = svm_load_model(
        os.path.join(
            config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
            'model_r_g2g.model'))
    p1_label, p1_acc, p1_val = svm_predict(y1, x1, model_p2p, '-q')
    acc1, _, _ = evaluations(y1, p1_label)
    p2_label, p2_acc, p2_val = svm_predict(y2, x2, model_p2G, '-q')
    acc2, _, _ = evaluations(y2, p2_label)
    p3_label, p3_acc, p3_val = svm_predict(y3, x3, model_G2G, '-q')
    acc3, _, _ = evaluations(y3, p3_label)
    with open('test.log', 'a') as f:
        print('{}, {}, {}'.format(acc1, acc2, acc3))
        f.write('{}, {}, {}\n'.format(acc1, acc2, acc3))


def save_ssvm_model(models, config):
    p2p_model_fn = os.path.join(
        config.get('REID', 'REWARD_MODEL_SAVED_PATH'), 'model_r_p2p.model')
    p2g_model_fn = os.path.join(
        config.get('REID', 'REWARD_MODEL_SAVED_PATH'), 'model_r_p2g.model')
    g2g_model_fn = os.path.join(
        config.get('REID', 'REWARD_MODEL_SAVED_PATH'), 'model_r_g2g.model')
    model_fns = [p2p_model_fn, p2g_model_fn, g2g_model_fn]

    for model, fn in zip(models, model_fns):
        with open(fn, 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read('/mnt/sdc1/wenhaoli/Learning-to-Group/code/config.ini')
    a = Dataset.identity_Dataset(config)
    train_album_list_fn = config.get('REID', 'TRAIN_ALBUM_LIST_FILE')
    a.loadAlbumList(train_album_list_fn)

    data = list(list())
    data.append([0, 0, 0])

    rb_p2p = ReplayBufferSimple(size=1000)
    rb_p2g = ReplayBufferSimple(size=1000)
    rb_g2g = ReplayBufferSimple(size=1000)
    rbs = [rb_p2p, rb_p2g, rb_g2g]
    f = frame.frame(rbs)
    if not os.path.exists(config.get('REID', 'REWARD_P2P_MODEL_TRAIN_DATA')):
        f.preTrainData(1000)
    f.init_replay_buffer_from_file()

    model_p2p = SGDClassifier(loss='modified_huber')
    model_p2g = SGDClassifier(loss='modified_huber')
    model_g2g = SGDClassifier(loss='modified_huber')
    models = [model_p2p, model_p2g, model_g2g]

    print('...train reward model...')
    start = time.time()
    for iteration in tqdm(range(1, 201)):
        # print '====================================================='
        # print 'Iter: %d' % iteration
        if iteration % 10 == 0:
            test_ssvm_model(config, models)
            save_ssvm_model(models, config)
            print('...ssvm model saved...')
        train_ssvm_model(config, f, models)
        # note that we don't have passerbys and don't care about them
        dataset = a.SimulateDataset(1000, 0.5, 0.5, s=20180724 + iteration)
        dataset.computeQuality()
        dataset.computeAffinity()
        f.reset()
        f.loadDataset(dataset)
        # model_p2p = svm_load_model(
        #     os.path.join(
        #         config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_p2p.model'))
        # model_p2G = svm_load_model(
        #     os.path.join(
        #         config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_p2g.model'))
        # model_G2G = svm_load_model(
        #     os.path.join(
        #         config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_g2g.model'))
        index = 1
        while f.checkState():
            package = f.getObservation()
            if type(package) == int:
                # print 'Done!'
                break
            data[0] = package
            question_type = len(package)
            if question_type == 3:  #point-----point
                # action, t1, t2 = svm_predict([0], data, model_p2p, '-q')
                action = model_p2p.predict(data)
                tp = 'P2P'
            elif question_type == 3 + f.k_size:  #point-----Group or group---point
                # action, t1, t2 = svm_predict([0], data, model_p2G, '-q')
                action = model_p2g.predict(data)
                tp = 'P2G'
            else:
                # action, t1, t2 = svm_predict([0], data, model_G2G, '-q')
                action = model_g2g.predict(data)
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
    end = time.time()
    print('reward model training finished, cost: {}s'.format(end - start))

    
