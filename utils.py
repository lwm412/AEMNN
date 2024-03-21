import shutil, os, csv, itertools, glob

import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim

from sklearn.metrics import confusion_matrix

import pandas as pd
import pickle as pk

# Utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious


def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()


def read_data(filename):
    print("Loading Data...")
    df = pd.read_csv(filename, header=None)
    data = df.values
    return data


def read_line(csvfile, line):
    with open(csvfile, 'r') as f:
        data = next(itertools.islice(csv.reader(f), line, None))
    return data


def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    target = target.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    conf_mat = confusion_matrix(pred.cpu().numpy(), target.data.cpu().numpy(), labels=range(2))
    return pred.cpu().numpy(), correct, target.size(0), conf_mat

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

"""数据加载器"""

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()

class StandardScaler:
    """标准转换器"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_csv(file_name):
    msisdn_list = []
    np_list = []
    label_list = []
    index = 0
    with open(file_name, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            msidn = row[0]
            daily = row[1:29]
            incre = row[29:57]
            seg = row[57:-2]
            fea = np.array(row[-2:]).astype(float)
            cur_label = int(fea[0])
            cur_ratio = float(fea[1])
            fea[cur_label] = cur_ratio
            fea[1 ^ cur_label] = 1 - cur_ratio
            d_array = np.array(daily).reshape(28, -1).astype(float)
            i_array = np.array(incre).reshape((28, -1)).astype(float)
            s_array = np.array(seg).reshape(28, -1).astype(float)
            assert (d_array.shape == (28, 1))
            assert (i_array.shape == (28, 1))
            assert (s_array.shape == (28, 12))

            np_list.append(np.concatenate((d_array, i_array, s_array), -1))
            label_list.append(fea.reshape(1, -1))
            msisdn_list.append(msidn)

    x_data = np.stack(np_list, 0)
    y_data = np.stack(label_list, 0)
    x_data = x_data.astype(float)
    y_data = y_data.astype(float)
    print(x_data.shape)
    print(y_data.shape)
    return x_data, y_data


def split_data(x_data, y_data):
    np.random.seed(123456)

    train_range = int(0.8 * x_data.shape[0])

    x_test = x_data[train_range:, :, :]
    y_test = y_data[train_range:, 0, :]

    x_rest = x_data[:train_range, :, :]
    y_rest = y_data[:train_range, 0, :]

    x_pos_valid_list = []
    y_pos_valid_list = []
    x_neg_valid_list = []
    y_neg_valid_list = []
    x_train_list = []
    y_train_list = []

    for i in range(x_rest.shape[0]):
        # print(y_rest[i, 0], y_rest[i, 1])
        if y_rest[i, 0] > y_rest[i, 1]:
            if len(x_neg_valid_list) < 800:
                x_neg_valid_list.append(x_rest[i, :, :])
                y_neg_valid_list.append(y_rest[i, :])
                continue
        else:
            if len(x_pos_valid_list) < 165:
                x_pos_valid_list.append(x_rest[i, :, :])
                y_pos_valid_list.append(y_rest[i, :])
                continue

        x_train_list.append(x_rest[i, :, :])
        y_train_list.append(y_rest[i, :])

    x_train = np.stack(x_train_list, axis=0)
    y_train = np.stack(y_train_list, axis=0)

    x_pos_valid_list.extend(x_neg_valid_list)
    y_pos_valid_list.extend(y_neg_valid_list)
    x_valid = np.stack(x_pos_valid_list, axis=0)
    y_valid = np.stack(y_pos_valid_list, axis=0)

    np.random.shuffle(x_train)
    np.random.shuffle(y_train)
    np.random.shuffle(x_valid)
    np.random.shuffle(y_valid)
    np.random.shuffle(x_test)
    np.random.shuffle(y_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_dataset(dataset_dir, normalizer, batch_size,
                 valid_batch_size=None, test_batch_size=None, column_wise=False,
                 train_ratio=0.7, test_ratio=0.2):
    """
    加载数据集
    :param train_ratio:
    :param test_ratio:
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    :param column_wise: 是指列元素的级别上进行归一，否则是全样本取值
    """
    data = {}
    x_data, y_data = load_csv(dataset_dir)
    train_range = int(0.8 * x_data.shape[0]) + 1

    # data["x_train"], data["y_train"] = x_data[:10601], y_data[:10601, 0, :]
    # data["x_test"], data["y_test"] = x_data[11601:], y_data[11601:, 0, :]
    # data["x_val"], data["y_val"] = data["x_test"], data["y_test"]

    data["x_train"], data["y_train"] = x_data[:train_range], y_data[:train_range, 0, :]
    data["x_test"], data["y_test"] = x_data[train_range:], y_data[train_range:, 0, :]
    data["x_val"], data["y_val"] = data["x_test"], data["y_test"]

    if normalizer == 0:
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 1:
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 2:
        if column_wise:
            mean = data['x_train'][..., 0].mean()  # 获得每列元素的均值、标准差
            std = data['x_train'][..., 0].std()
        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


