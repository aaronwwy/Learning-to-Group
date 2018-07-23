import os
import ConfigParser
from tqdm import tqdm
from sys import path
path.append('/media/deepglint/Data/Learning-to-Group/libsvm/python')
from svmutil import *
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


class ProfileDataset:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.imgfns = []
        self.features = []
        self.labels = []

    def format_labels(self):
        self.labels = [
            -1 if label.split(':')[-1].strip() == '2' else 1
            for label in self.labels
        ]

    def build(self):
        chunks = [
            f for f in os.listdir(self.data_folder)
            if os.path.isfile(os.path.join(self.data_folder, f))
        ]
        print('building dataset')
        for chunk in tqdm(chunks):
            chunk_fn = os.path.join(self.data_folder, chunk)
            fin = open(chunk_fn, 'r')
            text_data = fin.read().splitlines()
            for i in xrange(0, len(text_data) / 3):
                self.imgfns.append([text_data[i * 3]])
                self.labels.append(text_data[i * 3 + 1])
                self.features.append(map(float, text_data[i * 3 + 2].split()))
            fin.close()
        self.format_labels()

    def convert2libsvm(self, output):
        # if os.path.exists(output):
        #     print('converted file already exists, skip...')
        #     return
        if not os.path.exists(os.path.dirname(output)):
            print('{} is not exists, create it...'.format(
                os.path.dirname(output)))
            os.makedirs(os.path.dirname(output))
        n_outliers = 0
        with open(output, 'w') as f:
            print('creating libsvm format data')
            for j in tqdm(range(len(self.features))):
                if len(self.features[j]) != 128:
                    n_outliers += 1
                    continue
                f.write(" ".join([str(int(self.labels[j]))] + [
                    "{}:{}".format(i, self.features[j][i])
                    for i in range(len(self.features[0]))
                    if self.features[j][i] != 0
                ]))
                f.write('\n')
        print('n_outliers: {}'.format(n_outliers))

    @property
    def size(self):
        return len(self.labels)

    @property
    def n_negative(self):
        return (self.size - sum(self.labels)) / 2

    @property
    def n_positive(self):
        return self.size - self.n_negative


def train(config):
    start = time.time()

    param = svm_parameter('-h 0 -b 1')
    y, x = svm_read_problem(
        config.get('TEST', 'PROFILE_DATA_LIBSVM_FORMAT_SAVED_PATH'))
    problem = svm_problem(y, x)

    print('....training profile svm model')\

    model_saved_path = config.get('TEST', 'PROFILE_TRAINED_MODEL_PATH')
    if os.path.exists(model_saved_path):
        print('....model file already exists')
    else:
        model = svm_train(problem, param)
        model_dir = config.get('TEST', 'PROFILE_TRAINED_MODEL_DIR')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            svm_save_model(model_saved_path, model)

    end = time.time()

    print('....done, cost: {} s'.format(end - start))


def test(config):
    model_saved_path = config.get('TEST', 'PROFILE_TRAINED_MODEL_PATH')
    if not os.path.exists(model_saved_path):
        print('....you must train a model before test it')
        return
    model = svm_load_model(model_saved_path)

    data_path = config.get('TEST', 'PROFILE_DATA_LIBSVM_FORMAT_SAVED_PATH')
    if not os.path.exists(data_path):
        print('....test data does not exists')
        return
    y, x = svm_read_problem(data_path)
    p_label, p_acc, p_val = svm_predict(y, x, model, '-b 1')
    acc, mse, scc = evaluations(y, p_label)
    print(acc, mse, scc)


def virtualize_pesreid_quantity():
    config = ConfigParser.ConfigParser()
    config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')
    dataset = ProfileDataset(config.get('REID', 'PROFILE_DATA_FOLDER'))
    dataset.build()
    assert len(dataset.features) == len(dataset.labels) == len(
        dataset.imgfns) == dataset.size
    dataset.convert2libsvm(
        config.get('REID', 'PROFILE_DATA_LIBSVM_FORMAT_SAVED_PATH'))

    model_saved_path = config.get('TEST', 'PROFILE_TRAINED_MODEL_PATH')
    if not os.path.exists(model_saved_path):
        print('....you must train a model before test it')
        return
    model = svm_load_model(model_saved_path)

    data_path = config.get('REID', 'PROFILE_DATA_LIBSVM_FORMAT_SAVED_PATH')
    if not os.path.exists(data_path):
        print('....test data does not exists')
        return
    y, x = svm_read_problem(data_path)

    if os.path.exists(config.get('REID', 'PROFILE_CLASSIFIED_RESULT')):
        print('classified reid profile is already exists, skip...')
    else:
        print('inference start')
        s = time.time()
        p_label, p_acc, p_val = svm_predict(y, x, model, '-b 1 -q')
        e = time.time()
        print('done. cost: {}s'.format(e - s))

        with open(config.get('REID', 'PROFILE_CLASSIFIED_RESULT'), 'w') as f:
            for label, fn in zip(p_label, dataset.imgfns):
                f.write('{} {}\n'.format(label, fn))

    with open(config.get('REID', 'PROFILE_CLASSIFIED_RESULT'), 'r') as f:
        label_fn = [line.strip().split() for line in f.readlines()]

    neg_imgs = [rec[1] for rec in label_fn if float(rec[0]) == -1]
    pos_imgs = [rec[1] for rec in label_fn if float(rec[0]) == 1]

    # print(len(neg_imgs), len(pos_imgs))

    # random choice 3 images from pos and neg respectively
    for j in range(100):
        show_imgs = list(np.random.choice(neg_imgs, 10))
        show_imgs.extend(list(np.random.choice(pos_imgs, 10)))
        fig = plt.figure(figsize=(128, 64))
        columns = 10
        rows = 2
        for i in range(1, columns * rows + 1):
            img = show_imgs[i - 1][2:-2]
            fig.add_subplot(rows, columns, i)
            plt.imshow(mpimg.imread(img))
        plt.savefig('/media/deepglint/Data/Learning-to-Group/data/ReID/temp/{}.jpg'.format(j))
        plt.close()


def main():
    config = ConfigParser.ConfigParser()
    config.read('/media/deepglint/Data/Learning-to-Group/code/config.ini')
    dataset = ProfileDataset(config.get('TEST', 'PROFILE_DATA_FOLDER'))
    dataset.build()
    assert len(dataset.features) == len(dataset.labels) == len(
        dataset.imgfns) == dataset.size
    print(dataset.size, dataset.n_negative, dataset.n_positive)
    dataset.convert2libsvm(
        config.get('TEST', 'PROFILE_DATA_LIBSVM_FORMAT_SAVED_PATH'))
    train(config)
    test(config)


if __name__ == '__main__':
    virtualize_pesreid_quantity()