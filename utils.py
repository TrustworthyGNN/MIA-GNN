import pickle
import torch
import numpy as np
from keras import backend as K
from torch.utils.data import Dataset
import warnings
warnings.simplefilter("ignore")


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def load_pickled_data(path):
    with open(path, 'rb') as f:
        unPickler = pickle.load(f)
        return unPickler


def select_top_k(data, top=3):
    arr = []
    for d in data:
        top_k_idx = d.numpy().argsort()[::-1][0:top]
        # print(d.numpy()[top_k_idx])
        arr.append(d.numpy()[top_k_idx])
    return np.array(arr)
