#coding=utf-8#
import sys
import Dataset
import frame
from sys import path
path.append('libsvm/python')
from svmutil import *
import Evaluator
import Evaluate
import Dicision
import random
import numpy as np
import xgboost as xgb
import time
import os
import pickle
import copy


class test:
    def __init__(self, config, rbs, inference=False, output2file=False):
        self.Album = None
        self.dataset = None
        self.frame = None
        self.num = None
        self.Recall = 0
        self.Precision = 0
        self.Recall_edge = 0
        self.Precision_edge = 0
        self.operatenum = None
        self.dirname = None
        self.history = list()
        self.maxbatch = 20
        self.gamma = 0.9
        self.K = 5  #OP_k步
        self.beta = 0.5
        self.randchose = 0.5
        self.edge = list()
        self.result = list()
        self.accept = list()
        self.config = config
        self.inference = inference
        self.replay_buffers = rbs
        self.replay_buffer_p2p = self.replay_buffers[0]
        self.replay_buffer_p2g = self.replay_buffers[1]
        self.replay_buffer_g2g = self.replay_buffers[2]
        if output2file:
            self.output_method = self.output
        else:
            self.output_method = self.output_rb
        #0.9 0.6
    def init_replay_buffer_from_file(self):
        self.replay_buffer_p2p.add2(
            self.config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'))
        self.replay_buffer_p2g.add2(
            self.config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'))
        self.replay_buffer_g2g.add2(
            self.config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'))

    def reset(self, inference=False, output2file=False):
        self.Album = None
        self.dataset = None
        self.frame = None
        self.num = None
        self.Recall = 0
        self.Precision = 0
        self.Recall_edge = 0
        self.Precision_edge = 0
        self.operatenum = None
        self.dirname = None
        self.history = list()
        self.edge = list()
        self.result = list()
        self.accept = list()
        self.inference = inference
        if output2file:
            self.output_method = self.output
        else:
            self.output_method = self.output_rb

    def setbatch(self, batchsize):
        self.frame.trainbatch = batchsize

    def loadSimulate(self, dataset):
        f = frame.frame(training=False)
        f.loadDataset(dataset)
        self.frame = f
        self.dataset = dataset

    def dtOp(self, history, index):
        return history[index][-1] - history[index + self.K][-1]

    def QValue(self, history, index):
        if index == self.maxbatch - self.K - 1:
            return history[index][-2] + self.beta * self.dtOp(history, index)
        else:
            return history[index][-2] + self.beta * self.dtOp(
                history, index) + self.gamma * self.QValue(history, index + 1)

    def puthistory(self, feature, action, reward_action, Op, model):
        if len(self.history) >= self.maxbatch:
            #获取Q_value
            value_R = self.QValue(self.history, 0)
            if False:
                if model is None:
                    value_s_np1 = np.random.randn()
                else:
                    try:
                        sa = copy.copy(self.history[self.maxbatch - self.K][0])
                        sa.insert(0, self.history[self.maxbatch - self.K][1][0])
                        value_s_np1 = model.predict(xgb.DMatrix(np.array([sa])))[0]
                    except:
                        print(self.history[self.maxbatch - self.K][0])
                        print(self.history[self.maxbatch - self.K][1][0])
                        print(self.history[self.maxbatch - self.K][0].insert(
                            0, self.history[self.maxbatch - self.K][1][0]))
                        raise
                    value_s_np1 = value_s_np1 * (self.gamma**
                                                (self.maxbatch - self.K))
                value_R += value_s_np1
            #输出该记录
            self.output_method(self.history[0], value_R)
            #keep the length of history queue constant
            del self.history[0]
            self.history.append((feature, action, reward_action, Op))
        else:
            self.history.append((feature, action, reward_action, Op))

    def output(self, batch, value_R):
        if len(batch[0]) == 3:
            fout = open(self.config.get('REID', 'Q_P2P_MODEL_TRAIN_DATA'), 'a')
            fout.write(str(value_R))
            fout.write(' ' + '0:' + str(batch[1][0]))
            for i in xrange(0, 3):
                fout.write(' ' + str(i + 1) + ':' + str(batch[0][i]))
            fout.write('\n')
            fout.close()
        elif len(batch[0]) == 3 + self.frame.k_size:
            fout = open(self.config.get('REID', 'Q_P2G_MODEL_TRAIN_DATA'), 'a')
            fout.write(str(value_R))
            fout.write(' ' + '0:' + str(batch[1][0]))
            for i in xrange(0, 3 + self.frame.k_size):
                fout.write(' ' + str(i + 1) + ':' + str(batch[0][i]))
            fout.write('\n')
            fout.close()
        else:
            fout = open(self.config.get('REID', 'Q_G2G_MODEL_TRAIN_DATA'), 'a')
            fout.write(str(value_R))
            fout.write(' ' + '0:' + str(batch[1][0]))
            try:
                for i in xrange(0, 3 + 2 * self.frame.k_size):
                    fout.write(' ' + str(i + 1) + ':' + str(batch[0][i]))
            except:
                print(batch[0])
                print(len(batch[0]))
                print(3 + 2 * self.frame.k_size)
                raise
            fout.write('\n')
            fout.close()

    def output_rb(self, batch, value_R):
        if len(batch[0]) == 3:
            rec = str(value_R)
            rec += ' ' + '0:' + str(batch[1][0])
            for i in xrange(0, 3):
                rec += ' ' + str(i + 1) + ':' + str(batch[0][i])
            self.replay_buffer_p2p.add(rec)
        elif len(batch[0]) == 3 + self.frame.k_size:
            rec = str(value_R)
            rec += ' ' + '0:' + str(batch[1][0])
            for i in xrange(0, 3 + self.frame.k_size):
                rec += ' ' + str(i + 1) + ':' + str(batch[0][i])
            self.replay_buffer_p2g.add(rec)
        else:
            rec = str(value_R)
            rec += ' ' + '0:' + str(batch[1][0])
            for i in xrange(0, 3 + 2 * self.frame.k_size):
                rec += ' ' + str(i + 1) + ':' + str(batch[0][i])
            self.replay_buffer_g2g.add(rec)

    def begintest(self, iteration=0):
        # model_R_p2p = svm_load_model(
        #     os.path.join(
        #         self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_p2p.model'))
        # model_R_p2G = svm_load_model(
        #     os.path.join(
        #         self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_p2g.model'))
        # model_R_G2G = svm_load_model(
        #     os.path.join(
        #         self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
        #         'model_r_g2g.model'))
        with open(
                os.path.join(
                    self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
                    'model_r_p2p.model')) as f:
            model_R_p2p = pickle.load(f)
        with open(
                os.path.join(
                    self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
                    'model_r_p2g.model')) as f:
            model_R_p2G = pickle.load(f)
        with open(
                os.path.join(
                    self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
                    'model_r_g2g.model')) as f:
            model_R_G2G = pickle.load(f)

        is_first_iteration = False

        model_dir = self.config.get('REID', 'REWARD_MODEL_SAVED_PATH')
        if os.path.exists(os.path.join(model_dir, 'model_q_p2p.model')):
            model_Q_p2p = xgb.Booster(
                model_file=os.path.join(model_dir, 'model_q_p2p.model'))
            model_Q_p2G = xgb.Booster(
                model_file=os.path.join(model_dir, 'model_q_p2g.model'))
            model_Q_G2G = xgb.Booster(
                model_file=os.path.join(model_dir, 'model_q_g2g.model'))
        else:
            is_first_iteration = True

        data = list(list())
        data.append([0, 0, 0])
        data_Q = list(list())
        data_Q.append([0, 0, 0])
        index = 0
        reward = 0
        decision = Dicision.Dicision()
        t01 = time.time()

        while self.frame.checkState(check_batch=True):
            package = self.frame.getObservation()
            index += 1
            if type(package) == int:
                print 'Done!'
                break
            data[0] = package
            question_type = len(package)
            model = None
            if question_type == 3:  #point-----point
                if not is_first_iteration:
                    model = model_Q_p2p.copy()
                tp = 'P2P'
                #Reward Function
                # action_R, _, confidence = svm_predict([0], data, model_R_p2p,
                #                                       '-b 1 -q')
                # confidence = model_R_p2p.predict_proba(data)
                w = model_R_p2p.coef_[0]
                b = model_R_p2p.intercept_[0]
                #Reward Value Function: action = 0
                temp = package[:]
                temp.insert(0, 0)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_0 = model_Q_p2p.predict(DM_data)
                else:
                    value_0 = [random.random()]
                del temp[0]
                #Reward Value Function: action = 1
                temp.insert(0, 1)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_1 = model_Q_p2p.predict(DM_data)
                else:
                    value_1 = [random.random()]
                #choose the most awarded action
                if value_1[0] >= value_0[0]:
                    action = [1]
                else:
                    action = [0]

            elif question_type == 3 + self.frame.k_size:  #point-----Group or group---point
                if not is_first_iteration:
                    model = model_Q_p2G.copy()
                tp = 'P2G'
                #Reward Function
                # action_R, _, confidence = svm_predict([0], data, model_R_p2G,
                #                                       '-b 1 -q')
                # confidence = model_R_p2G.predict_proba(data)
                w = model_R_p2G.coef_[0]
                b = model_R_p2G.intercept_[0]
                #Reward Value Function: action = 0
                temp = package[:]
                temp.insert(0, 0)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_0 = model_Q_p2G.predict(DM_data)
                else:
                    value_0 = [random.random()]
                del temp[0]
                #Reward Value Function: action = 1
                temp.insert(0, 1)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_1 = model_Q_p2G.predict(DM_data)
                else:
                    value_1 = [random.random()]
                #choose the most awarded action
                if value_1[0] >= value_0[0]:
                    action = [1]
                else:
                    action = [0]
            else:
                if not is_first_iteration:
                    model = model_Q_G2G.copy()
                tp = 'G2G'
                #Reward Function
                # action_R, _, confidence = svm_predict([0], data, model_R_G2G,
                #                                       '-b 1 -q')
                # confidence = model_R_G2G.predict_proba(data)
                w = model_R_G2G.coef_[0]
                b = model_R_G2G.intercept_[0]
                #Reward Value Function: action = 0
                temp = package[:]
                temp.insert(0, 0)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_0 = model_Q_G2G.predict(DM_data)
                else:
                    value_0 = [random.random()]
                del temp[0]
                #Reward Value Function: action = 1
                temp.insert(0, 1)
                data_Q[0] = temp
                DM_data = xgb.DMatrix(np.array(data_Q))
                if not is_first_iteration:
                    value_1 = model_Q_G2G.predict(DM_data)
                else:
                    value_1 = [random.random()]
                #choose the most awarded action
                if value_1[0] > value_0[0]:
                    action = [1]
                else:
                    action = [0]
            #获取操作量原数量
            # t-lambda processing  (iteration in [1,400])
            if random.random() >= (0.025 * iteration):
                action = [random.randint(0, 1)]

            # get reward of the action
            # reward_action = 10 * abs(2 * confidence[0] - 1)
            reward_action = abs(np.sum(np.multiply(w, package)) + b)

            #get the variance of operate number
            self.frame.Normalize_label()
            operatenum_pre = Evaluator.evaluate(self.dataset.imgID,
                                                self.frame.label, [0])

            #check the action is True or False
            action_result = self.frame.setPerception(action, save=False)
            if action_result == False:
                reward_action = -reward_action
            #save history
            self.puthistory(package, action, reward_action, operatenum_pre,
                            model)

        if not self.inference:
            #calculate Metric
            self.frame.Normalize_label()
            self.Recall = Evaluate.Recall(self.dataset.imgID, self.frame.label)
            self.Precision = Evaluate.Precision(self.dataset.imgID,
                                                self.frame.label)
            self.operatenum = Evaluator.evaluate(self.dataset.imgID,
                                                 self.frame.label, [0])
            self.Recall_edge = Evaluate.Recall_edge(self.dataset.imgID,
                                                    self.frame.label, 0)
            self.Precision_edge = Evaluate.Precision_edge(
                self.dataset.imgID, self.frame.label)
            print self.dataset.size, self.Recall_edge, self.Precision_edge, self.operatenum
            with open(
                    os.path.join(
                        self.config.get('REID', 'REWARD_MODEL_SAVED_PATH'),
                        'xgboost_output_nstepsarsa_origin.log'), 'a') as f:
                f.write('{}, {}, {}, {}\n'.format(
                    self.dataset.size, self.Recall_edge, self.Precision_edge,
                    self.operatenum))


if __name__ == '__main__':
    t = test()
