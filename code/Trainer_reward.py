#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import reward_value_test
import xgboost as xgb
import load_test_data
import traceback
import ConfigParser
import os
import shutil
import pickle
import time
from tqdm import tqdm
import numpy as np
from replay_buffer import ReplayBufferSimple
'''
reward Q-value and value-function 
'''


def trainSVMModel(config):
    print 'Train SVM model start'
    param = svm_parameter('-s 0 -t 0 -b 1')
    y1, x1 = svm_read_problem(
        config.get('REID', 'REWARD_P2P_MODEL_TRAIN_DATA'))
    y2, x2 = svm_read_problem(
        config.get('REID', 'REWARD_P2G_MODEL_TRAIN_DATA'))
    y3, x3 = svm_read_problem(
        config.get('REID', 'REWARD_G2G_MODEL_TRAIN_DATA'))
    prob1 = svm_problem(y1, x1)
    prob2 = svm_problem(y2, x2)
    prob3 = svm_problem(y3, x3)
    print '....training p2p'
    model_p2p = svm_train(prob1, param)
    print '....training p2G'
    model_p2g = svm_train(prob2, param)
    print '....training G2G'
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
    print 'Train new model finished'


def trainXGBmodel(machine, config):
    print 'Train XGBoost model start'
    machine.replay_buffer_p2p.sample2(
        config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'), 1000)
    machine.replay_buffer_p2g.sample2(
        config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'), 1000)
    machine.replay_buffer_g2g.sample2(
        config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'), 1000)
    dtrain_p2p = xgb.DMatrix(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
    dtrain_p2G = xgb.DMatrix(config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'))
    dtrain_G2G = xgb.DMatrix(config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'))
    param = {'max_depth': 5, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    model_dir = config.get('REID', 'REWARD_MODEL_SAVED_PATH')
    numround = 100
    if not os.path.exists(os.path.join(model_dir, 'model_q_p2p.model')):
        bst_p2p = xgb.train(param, dtrain_p2p, numround)
        bst_p2G = xgb.train(param, dtrain_p2G, numround)
        bst_G2G = xgb.train(param, dtrain_G2G, numround)
    else:
        bst_p2p = xgb.train(
            param,
            dtrain_p2p,
            numround,
            xgb_model=os.path.join(model_dir, 'model_q_p2p.model'))
        bst_p2G = xgb.train(
            param,
            dtrain_p2G,
            numround,
            xgb_model=os.path.join(model_dir, 'model_q_p2g.model'))
        bst_G2G = xgb.train(
            param,
            dtrain_G2G,
            numround,
            xgb_model=os.path.join(model_dir, 'model_q_g2g.model'))
    bst_p2p.save_model(os.path.join(model_dir, 'model_q_p2p.model'))
    bst_p2G.save_model(os.path.join(model_dir, 'model_q_p2g.model'))
    bst_G2G.save_model(os.path.join(model_dir, 'model_q_g2g.model'))
    print 'Train XGBoost model finished'


def reward_normalize(path):
    norm_line = []
    rewards = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(' ')
            rewards.append(float(data[0]))
            norm_line.append(data)
    
    r_mean = np.mean(rewards)
    r_var = np.var(rewards)

    with open(path, 'w') as f:
        for data in norm_line:
            data[0] = str((float(data[0]) - r_mean) / (r_var + 1e-6))
            f.write(' '.join(data))
            f.write('\n')


def trainXGBmodel_v2(config):
    print 'Train XGBoost model start'
    param = {'max_depth': 5, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    model_dir = config.get('REID', 'REWARD_MODEL_SAVED_PATH')
    numround = 100

    if os.path.exists(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA')):
        reward_normalize(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
        dtrain_p2p = xgb.DMatrix(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
        if not os.path.exists(os.path.join(model_dir, 'model_q_p2p.model')):
            bst_p2p = xgb.train(param, dtrain_p2p, numround)
        else:
            bst_p2p = xgb.train(
                param,
                dtrain_p2p,
                numround,
                xgb_model=os.path.join(model_dir, 'model_q_p2p.model'))
        bst_p2p.save_model(os.path.join(model_dir, 'model_q_p2p.model'))

    if os.path.exists(config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA')):
        dtrain_p2g = xgb.DMatrix(config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'))
        if not os.path.exists(os.path.join(model_dir, 'model_q_p2g.model')):
            bst_p2g = xgb.train(param, dtrain_p2g, numround)
        else:
            bst_p2g = xgb.train(
                param,
                dtrain_p2g,
                numround,
                xgb_model=os.path.join(model_dir, 'model_q_p2g.model'))
        bst_p2g.save_model(os.path.join(model_dir, 'model_q_p2g.model'))

    if os.path.exists(config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA')):
        dtrain_g2g = xgb.DMatrix(config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'))
        if not os.path.exists(os.path.join(model_dir, 'model_q_g2g.model')):
            bst_g2g = xgb.train(param, dtrain_g2g, numround)
        else:
            bst_g2g = xgb.train(
                param,
                dtrain_g2g,
                numround,
                xgb_model=os.path.join(model_dir, 'model_q_g2g.model'))
        bst_g2g.save_model(os.path.join(model_dir, 'model_q_g2g.model'))
    print 'Train XGBoost model finished'


def collect_start_data(machine, idset, config, n):
    print('...collect start data...')
    batchsize = 100
    for _ in tqdm(range(10)):
        dataset = idset.SimulateDataset(1000, 1.0, 0.0, s=20180725 - n * 100)
        #dataset=load_test_data., fd_LFW_dataset(filepath)
        dataset.computeQuality()
        dataset.computeAffinity()
        #do this Dataset
        machine.reset(inference=True, output2file=True)
        machine.loadSimulate(dataset)
        machine.setbatch(batchsize)
        # try:
        machine.begintest(1000)


def testXGBmodel(machine, config):
    print('...test model...')
    with open(config.get('REID', 'XGBOOST_TEST_DATASET'), 'rb') as f:
        test_dataset = pickle.load(f)
        machine.reset(inference=False, output2file=True)
        machine.loadSimulate(test_dataset)
        machine.setbatch(4)
        machine.begintest(1000)


def clean(config, data=True, model=True):
    if data:
        if os.path.exists(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA')):
            try:
                os.remove(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
                os.remove(config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'))
                os.remove(config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'))
            except:
                pass
    if model:
        model_dir = config.get('REID', 'REWARD_MODEL_SAVED_PATH')
        if os.path.exists(os.path.join(model_dir, 'model_q_p2p.model')):
            try:
                os.remove(os.path.join(model_dir, 'model_q_p2p.model'))
                os.remove(os.path.join(model_dir, 'model_q_p2g.model'))
                os.remove(os.path.join(model_dir, 'model_q_g2g.model'))
            except:
                pass


def main():
    config = ConfigParser.ConfigParser()
    config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')
    clean(config)
    a = Dataset.identity_Dataset(config)
    train_album_list_fn = config.get('REID', 'TRAIN_ALBUM_LIST_FILE')
    a.loadAlbumList(train_album_list_fn)
    #fin_train=open('data/train_LFW_B','r')
    #trainlist=fin_train.read().splitlines()
    #trainlist=trainlist*10
    data = list(list())
    data.append([0, 0, 0])
    iteration = 1
    batchsize = 100

    rb_p2p_q = ReplayBufferSimple(size=10000)
    rb_p2g_q = ReplayBufferSimple(size=10000)
    rb_g2g_q = ReplayBufferSimple(size=10000)
    rbs_q = [rb_p2p_q, rb_p2g_q, rb_g2g_q]

    machine = reward_value_test.test(
        config, rbs_q, inference=True, output2file=True)

    if False:
        continue_collect_data = True
        for i in range(1, 10):
            collect_start_data(machine, a, config, n=i)
            dtrain_p2p_fn = config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA')
            dtrain_p2G_fn = config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA')
            dtrain_G2G_fn = config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA')
            if os.path.exists(dtrain_p2p_fn) and os.path.exists(
                    dtrain_p2G_fn) and os.path.exists(dtrain_G2G_fn):
                continue_collect_data = False
            if not continue_collect_data:
                break
            else:
                print('...start data not collect finish, continue...')
        machine.init_replay_buffer_from_file()
    # create test file
    test_dataset = a.SimulateDataset(1000, 1.0, 0.0, s=31415926)
    test_dataset.computeQuality()
    test_dataset.computeAffinity()
    with open(config.get('REID', 'XGBOOST_TEST_DATASET'), 'wb') as f:
        pickle.dump(test_dataset, f)

    print('...training start...')
    start = time.time()
    while iteration < 500:
        #for filepath in trainlist:
        print '====================================================='
        print 'Iter: %d' % iteration
        t1 = time.time()
        dataset = a.SimulateDataset(1000, 1.0, 0.0, s=20180725 + iteration)
        t2 = time.time()
        #dataset=load_test_data., fd_LFW_dataset(filepath)
        dataset.computeQuality()
        t3 = time.time()
        dataset.computeAffinity()
        #do this Dataset
        t4 = time.time()
        machine.reset(inference=False, output2file=True)
        t5 = time.time()
        clean(config, model=False)
        t6 = time.time()
        machine.loadSimulate(dataset)
        t7 = time.time()
        machine.setbatch(batchsize)
        # try:
        t8 = time.time()
        machine.begintest(iteration - 1)
        # except:
        #     f = open('./log/log.txt', 'a')
        #     traceback.print_exc(file=f)
        #     f.write('=' * 20)
        #     f.write('\n')
        #     f.flush()
        #     f.close()

        batchsize += 1
        #训练新模型
        # if iteration % 100 == 0:
        # trainSVMModel(config)
        t9 = time.time()
        trainXGBmodel_v2(config)
        t10 = time.time()
        if iteration % 10 == 0:
            testXGBmodel(machine, config)
        # print machine.operatenum
        with open(
                '/media/deepglint/Data/Learning-to-Group/model/ReID/time.log',
                'a') as f:
            f.write(' '.join(
                map(str, [
                    t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6,
                    t8 - t7, t9 - t8, t10 - t9
                ])))
            f.write('\n')

        iteration += 1
    end = time.time()
    print('Done, time costs: {} h'.format((end - start) / 3600))


if __name__ == '__main__':
    main()