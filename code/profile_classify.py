import os
import ConfigParser
from tqdm import tqdm
from sys import path
path.append('libsvm/python')
from svmutil import *
import time


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
        n_outliers = 0
        with open(output, 'w') as f:
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
        print(n_outliers)

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


if __name__ == '__main__':
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