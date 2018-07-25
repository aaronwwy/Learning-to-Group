#coding=utf-8#
import numpy as np
import random
from scipy import spatial as spt
import sys
import math
import os
import shutil
from sys import path
path.append('libsvm/python')
from svmutil import *
from tqdm import tqdm
import ConfigParser


def format(config):
    '''
    1. Extract all profile image from each album to form a new album,
       and delete these image in source album at the same time.
    2. Create feature files for each people
    '''
    feature_dir = config.get('TEST', 'PROFILE_DATA_FOLDER')
    formatted_feature_dir = config.get('TEST', 'FORMATTED_PROFILE_DATA_FOLDER')
    if not os.path.exists(formatted_feature_dir):
        os.makedirs(formatted_feature_dir)
    else:
        print('...formatted feature files already exist, skip...')
        return

    feature_files = [
        f for f in os.listdir(feature_dir)
        if os.path.isfile(os.path.join(feature_dir, f))
    ]
    profile_text_data = []
    passerby_text_data = []

    pid_dict = {}
    profile_dict = {}
    passerby_dict = {}

    for feature_file in tqdm(feature_files):
        feature_fn = os.path.join(feature_dir, feature_file)

        with open(feature_fn, 'r') as fin:
            text_data = [line.strip() for line in fin.readlines()]
            new_text_data = []

            for i in xrange(0, len(text_data) / 3):
                if len(text_data[i * 3 + 2].split(' ')) != 128:
                    print('...get a outlier, skip...')
                    continue
                flag = text_data[i * 3 + 1].split(':')[-1].strip()
                is_profile = (flag == '2')
                is_passerby = (flag == '1')
                if is_profile and text_data[i * 3] not in profile_dict.keys():
                    profile_dict[text_data[i * 3]] = 0
                    profile_text_data.extend(text_data[i * 3:i * 3 + 3])
                elif is_passerby and text_data[i *
                                               3] not in passerby_dict.keys():
                    passerby_dict[text_data[i * 3]] = 0
                    passerby_text_data.extend(text_data[i * 3:i * 3 + 3])
                else:
                    pid = '-'.join(text_data[i * 3].split('/')[1:3])
                    pid_iid = text_data[i * 3].split('/')[3]
                    if pid not in pid_dict.keys():
                        pid_dict[pid] = {}
                    else:
                        pid_dict[pid][pid_iid] = text_data[i * 3:i * 3 + 3]

    total_pid = 3
    for pid, feat_dic in pid_dict.items():
        with open(
                os.path.join(formatted_feature_dir,
                             str(total_pid) + '.txt'), 'a') as f:
            for pid_iid, feat in feat_dic.items():
                feat[1] = 'ID: {}'.format(total_pid)
                f.write('\n'.join(feat))
                f.write('\n')
        total_pid += 1

    with open(os.path.join(formatted_feature_dir, '2.txt'), 'w') as f:
        f.write('\n'.join(profile_text_data))
    with open(os.path.join(formatted_feature_dir, '1.txt'), 'w') as f:
        f.write('\n'.join(passerby_text_data))


def create_train_album_list(config):
    album_list_fn = config.get('TEST', 'TRAIN_ALBUM_LIST_FILE')
    if os.path.exists(album_list_fn):
        print('...train album list file already exists, skip...')
        return

    formatted_feature_dir = config.get('TEST', 'FORMATTED_PROFILE_DATA_FOLDER')
    album_list = [
        f for f in os.listdir(formatted_feature_dir)
        if f != '2.txt' and f != '1.txt'
    ]
    album_list.insert(0, '2.txt')
    album_list.insert(0, '1.txt')

    with open(album_list_fn, 'w') as f:
        f.write('\n'.join([
            os.path.join(formatted_feature_dir, album) for album in album_list
        ]))

    print('...create train album list file done...')


def create_reid_album_list(config):
    album_list_fn = config.get('REID', 'ALBUM_LIST_FILE')
    album_train_list_fn = config.get('REID', 'TRAIN_ALBUM_LIST_FILE')
    album_test_list_fn = config.get('REID', 'TEST_ALBUM_LIST_FILE')
    if os.path.exists(album_list_fn):
        print('...album list file already exists, skip...')
        return

    formatted_feature_dir = config.get('REID', 'FORMATTED_PROFILE_DATA_FOLDER')
    album_list = [
        f for f in os.listdir(formatted_feature_dir) if '-1' not in f
    ]

    np.random.seed(20180724)
    train_prop = float(config.get('REID', 'TRAIN_ALBUM_PROPORTION'))
    album_train_list = np.random.choice(
        album_list, int(train_prop * len(album_list)), replace=False)
    album_test_list = [a for a in album_list if a not in album_train_list]
    with open(album_list_fn, 'w') as alf:
        alf.write(os.path.join(formatted_feature_dir, '-1.txt'))
        alf.write('\n')
        alf.write('\n'.join([
            os.path.join(formatted_feature_dir, album) for album in album_list
        ]))
        print('...create reid album list file done...')
    with open(album_train_list_fn, 'w') as atrlf:
        atrlf.write(os.path.join(formatted_feature_dir, '-1.txt'))
        atrlf.write('\n')
        atrlf.write('\n'.join([
            os.path.join(formatted_feature_dir, album) for album in album_list
        ]))
        print('...create reid train album list file done...')
    with open(album_test_list_fn, 'w') as atelf:
        atelf.write(os.path.join(formatted_feature_dir, '-1.txt'))
        atelf.write('\n')
        atelf.write('\n'.join([
            os.path.join(formatted_feature_dir, album) for album in album_list
        ]))
        print('...create reid test album list file done...')


class Dataset:
    def __init__(self, config):
        self.datasetID = 0
        self.imageNameList = list()
        self.rect = list(list())
        self.feature = list(list())
        self.imgID = list()
        self.Affinity = None
        self.Quality = None  #正脸质量,越低侧脸概率越高
        self.albumnum = None  #类别数量
        self.size = 0
        self.isnoise = None  #判断是否是噪音，as第一层分类器
        self.config = config

    def loadfeature(self, featurefileName):
        # print 'Load %s' % featurefileName
        fin = open(featurefileName, 'r')
        text_data = fin.read().splitlines()
        #训练用格式：
        for i in tqdm(xrange(0, len(text_data) / 3)):
            self.imageNameList.append([text_data[i * 3]])
            self.feature.append(map(float, text_data[i * 3 + 2].split()))
            if len(self.feature[-1]) != 128:
                print(len(self.feature[-1]))
                print(featurefileName)
                print(text_data[i * 3])
                # assert False
            # just fill something
            self.imgID.append(0)
            self.size += 1
        '''
        for i in xrange(0, len(text_data) / 3):
            self.imageNameList.append([text_data[i * 3]])
            self.rect.append(map(int, text_data[i * 3 + 1].split()))
            self.feature.append(map(float, text_data[i * 3 + 2].split()))
            self.imgID.append(0)
            self.size += 1
            # print i
        '''
        fin.close()

    def computeAffinity(self):
        # print "start computeAffinity..."
        try:
            self.Affinity = 1 - spt.distance.pdist(self.feature, 'cosine')
        except ValueError as e:
            print(np.array(self.feature).shape)
            # print(self.feature[0])
            raise
        print "Compute Affinity finished"

    def computeQuality(self):
        # print 'Compute Quality start'
        model_saved_path = self.config.get('REID',
                                           'PROFILE_TRAINED_MODEL_PATH')
        model = svm_load_model(model_saved_path)
        #model=svm_load_model('model/model_profile_300.model')
        # print(len(self.imgID), self.size)
        p_label, p_acc, p_vals = svm_predict(
            [-1 if self.imgID[i] == 0 else 1 for i in range(0, self.size)],
            self.feature, model, '-b 1 -q')
        #    p_label_b,p_acc_b,p_vals_b=svm_predict([self.imgID[i]==2 for i in range(0,self.size)],self.feature,model_b)
        self.Quality = [x[1] for x in p_vals]
        #self.Quality=[1 for x in xrange(self.size)]
        print 'Compute Quality finished'


class identity_Dataset:
    def __init__(self, config):
        self.album = list()
        self.albumCount = 0
        self.config = config

    def loadAlbumList(self, albumlistname):
        fin = open(albumlistname, 'r')
        text_data = fin.read().splitlines()
        id = 0
        print('...load album list...')
        for filepath in tqdm(text_data):
            temp = Dataset(self.config)
            temp.loadfeature(filepath)
            temp.datasetID = id
            id = id + 1
            self.albumCount += 1
            self.album.append(temp)
        fin.close()

    #identity_ratio+profile_ratio < 1.0
    def SimulateDataset(self, albumsize, identity_ratio, profile_ratio):
        # random.seed(20180718)
        dataset = Dataset(self.config)
        albumnum = 0
        #load identity_image
        identity_size = albumsize * identity_ratio
        identity_num = 0
        album_shuffle = range(1, self.albumCount)
        random.shuffle(album_shuffle)
        for identity_index in album_shuffle:  #xrange(2,self.albumCount):
            albumnum += 1
            for i in xrange(0, self.album[identity_index].size):
                dataset.imageNameList.append(
                    self.album[identity_index].imageNameList[i])
                # dataset.rect.append(self.album[identity_index].rect[i])
                dataset.feature.append(self.album[identity_index].feature[i])
                dataset.imgID.append(self.album[identity_index].datasetID)
                identity_num += 1
                if identity_num >= identity_size:
                    break
            if identity_num >= identity_size:
                break

        #load profile_image
        profile_size = albumsize * profile_ratio
        profile_num = 0
        profile_index = 0
        album_shuffle = range(0, self.album[profile_index].size)
        random.shuffle(album_shuffle)
        albumnum += 1
        for i in album_shuffle:
            if profile_num >= profile_size:
                break
            dataset.imageNameList.append(
                self.album[profile_index].imageNameList[i])
            # dataset.rect.append(self.album[profile_index].rect[i])
            dataset.feature.append(self.album[profile_index].feature[i])
            dataset.imgID.append(self.album[profile_index].datasetID)
            profile_num += 1

        #load passerby_image
        # passerby_size = albumsize - profile_size - identity_size
        # passerby_num = 0
        # passerby_index = 0
        # album_shuffle = range(0, self.album[passerby_index].size)
        # random.shuffle(album_shuffle)
        # if passerby_size > 0:
        #     albumnum += 1
        # for i in album_shuffle:
        #     if passerby_num >= passerby_size:
        #         break
        #     dataset.imageNameList.append(
        #         self.album[passerby_index].imageNameList[i])
        #     # dataset.rect.append(self.album[passerby_index].rect[i])
        #     dataset.feature.append(self.album[passerby_index].feature[i])
        #     dataset.imgID.append(self.album[passerby_index].datasetID)
        #     passerby_num += 1

        dataset.albumnum = albumnum
        dataset.size = len(dataset.imgID)
        return dataset


if __name__ == '__main__':
    for _ in range(1):
        config = ConfigParser.ConfigParser()
        config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')

        # reid dateset already formatted
        # format(config)
        create_reid_album_list(config)

        b = identity_Dataset(config)
        album_list_fn = config.get('REID', 'TRAIN_ALBUM_LIST_FILE')
        b.loadAlbumList(album_list_fn)
        c = b.SimulateDataset(1000, 0.5, 0.5)

        c.computeAffinity()
        c.computeQuality()
