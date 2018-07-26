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
from tqdm import tqdm
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


def trainXGBmodel(config):
    print 'Train XGBoost model start'
    dtrain_p2p = xgb.DMatrix(config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
    dtrain_p2G = xgb.DMatrix(config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'))
    dtrain_G2G = xgb.DMatrix(config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'))
    param = {'max_depth': 5, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    numround = 100
    bst_p2p = xgb.train(param, dtrain_p2p, numround)
    bst_p2G = xgb.train(param, dtrain_p2G, numround)
    bst_G2G = xgb.train(param, dtrain_G2G, numround)
    model_dir = config.get('REID', 'REWARD_MODEL_SAVED_PATH')
    bst_p2p.save_model(os.path.join(model_dir, 'model_q_p2p.model'))
    bst_p2G.save_model(os.path.join(model_dir, 'model_q_p2g.model'))
    bst_G2G.save_model(os.path.join(model_dir, 'model_q_g2g.model'))
    print 'Train XGBoost model finished'


def collect_start_date(idset, config, n):
    print('...collect start data...')
    batchsize = 4
    for iteration in tqdm(range(100)):
        dataset = idset.SimulateDataset(1000, 0.5, 0.5, s=20180725 - n * 100)
        #dataset=load_test_data.load_LFW_dataset(filepath)
        dataset.computeQuality()
        dataset.computeAffinity()
        #do this Dataset
        machine = reward_value_test.test(config, inference=True)
        machine.loadSimulate(dataset)
        machine.setbatch(batchsize)
        machine.begintest(iteration - 1)
        batchsize += 1


def main():
    config = ConfigParser.ConfigParser()
    config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')
    a = Dataset.identity_Dataset(config)
    train_album_list_fn = config.get('REID', 'TRAIN_ALBUM_LIST_FILE')
    a.loadAlbumList(train_album_list_fn)
    #fin_train=open('data/train_LFW_B','r')
    #trainlist=fin_train.read().splitlines()
    #trainlist=trainlist*10
    data = list(list())
    data.append([0, 0, 0])
    iteration = 1
    batchsize = 4

    continue_collect_data = True
    for i in range(1, 10):
        collect_start_date(a, config, n=i)
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

    print('...training start...')
    while iteration < 1001:
        #for filepath in trainlist:
        print '====================================================='
        print 'Iter: %d' % iteration
        dataset = a.SimulateDataset(1000, 0.5, 0.5, s=20180725 + iteration)
        #dataset=load_test_data.load_LFW_dataset(filepath)
        dataset.computeQuality()
        dataset.computeAffinity()
        #do this Dataset
        machine = reward_value_test.test(config)
        machine.loadSimulate(dataset)
        machine.setbatch(batchsize)
        # try:
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
        if iteration % 100 == 0:
            trainSVMModel(config)
        trainXGBmodel(config)
        # print machine.operatenum

        iteration += 1

    print 'Done'


if __name__ == '__main__':
    main()