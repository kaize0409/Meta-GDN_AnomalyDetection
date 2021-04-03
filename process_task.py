import numpy as np
import pandas as pd
import scipy.io as sio
from data import *
import torch
import glob
import random


def task_process():
    '''
    :return: array of size [nb_tasks, network_size, ]
    '''
    pass

def remove_values(arr1, arr2):

    res = [e for e in arr1 if e not in arr2]
    return np.array(res)

def main():
    pass

class task:

    def __init__(self, nb_task, degree, t_ratio, l_ratio, name):
        self.nb_task = nb_task
        self.degree = degree
        self.tr = t_ratio
        self.lr = l_ratio
        self.feature_l = []
        self.label_l = []
        self.label = None
        self.labeled_l = []
        self.unlabeled_l = []
        self.f_name = []
        self.target = None
        self.name = name
    def loadNProcess(self):
        if self.name == "yelp":
            l = sorted(glob.glob('graphs/*.mat'))
        elif self.name == "pubmed":
            l = glob.glob("data/Pubmed/*.mat")
        elif self.name == "reddit":
            l = glob.glob("data/reddit/*.mat")
        else:
            l = []
        # f_l = list(l[i] for i in [0, 1, 4, 5, 7])
        f_l = random.sample(l, self.nb_task)
        print('Filenames: ', ', '.join(f_l))
        random.shuffle(f_l)
        for f in f_l:
            adj, feature, label = load_yelp(f)
            # self.adj_l.append(adj)
            adj = normalize_adjacency(adj)
            feature = normalize_feature(feature)
            adj = sp_matrix_to_torch_sparse_tensor(adj).float()
            feature = torch.FloatTensor(feature.toarray())
            for i in range(self.degree):
                feature = torch.sparse.mm(adj, feature)

            self.feature_l.append(feature)
            self.label_l.append(label)
            self.target = f
            self.f_name.append(f[7:-4])
        print("data loading finished.")

    def sampleAnomaly(self):
        # sampling anomalies from training task
        for i in range(self.nb_task - 1):
            # print("in %d-th task" % i)
            label_tmp = self.label_l[i]
            idx_anomaly = np.nonzero(label_tmp == 1)[0]
            idx_normal = np.nonzero(label_tmp == 0)[0]
            nb_ano = int(len(idx_anomaly)*self.lr)
            nb_ano = 10
            # print(nb_ano)
            idx_labeled = np.random.choice(idx_anomaly, size=nb_ano, replace=False)
            # print(len(idx_labeled))
            self.labeled_l.append(idx_labeled)
            idx_unlabeled = remove_values(idx_anomaly, idx_labeled)
            self.unlabeled_l.append(np.concatenate((idx_normal, idx_unlabeled)).tolist())

        label_tmp = self.label_l[self.nb_task - 1]
        idx_anomaly = np.nonzero(label_tmp == 1)[0]
        idx_normal = np.nonzero(label_tmp == 0)[0]
        np.random.shuffle(idx_normal)
        np.random.shuffle(idx_anomaly)
        [ano_train, ano_test] = np.array_split(idx_anomaly, [int(self.tr * len(idx_anomaly))])
        [nor_train, nor_test] = np.array_split(idx_normal, [int(self.tr * len(idx_normal))])
        idx_test = np.concatenate((ano_test, nor_test)).tolist()
        nb = int(len(idx_anomaly) * self.lr)
        nb = 10
        idx_labeled = np.random.choice(ano_train, size=nb, replace=False)
        self.labeled_l.append(idx_labeled)
        idx_unlabeled = remove_values(ano_train, idx_labeled)
        self.unlabeled_l.append(np.concatenate((nor_train, idx_unlabeled)).tolist())

        # for i in range(self.nb_task):
        #     self.label_l[i] = torch.FloatTensor(self.label_l[i])
        self.label = torch.FloatTensor(label_tmp)
        return self.feature_l, self.label, self.labeled_l, self.unlabeled_l, idx_test

def process_Yelp(nb_task,f_l=None,degree=2,tr1=0.7,tr2=0.4,lr=0.08):
    if f_l is None:
        f_l = glob.glob('graphs/*.mat')
    print(len(f_l))
    file_l = random.sample(f_l, nb_task)
    for f in file_l[:-1]:
        adj, feature, label = load_yelp(f)
        idx_anomaly = np.nonzero(label == 1)[0]
        idx_normal = np.nonzero(label == 0)[0]
        print(len(idx_anomaly))
        print(len(idx_normal))


if __name__ == "__main__":
    main()
    ins = task(nb_task=2,degree=2)
    ins.loadNProcess()
    a,b,c,d,e = ins.sampleAnomaly()
