import scipy.sparse as sp
import scipy.io as sio
import pandas as pd
import torch
import numpy as np
import networkx as nx
import csv
import sys
import pickle as pkl
from datetime import date, datetime
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components



class DataLoader:
    def __init__(self, features, idx_l, idx_u, b_size):
        self.features = features
        # self.label = label
        self.idx_labeled = idx_l
        self.idx_unlabeled = idx_u
        self.bs = b_size

    def getBatch(self):
        # idx_x = []
        y_batch = []
        idx_x = np.random.choice(self.idx_labeled, size=int(self.bs / 2), replace=False).tolist()
        idx_x += np.random.choice(self.idx_unlabeled, size=int(self.bs / 2), replace=False).tolist()
        y_batch = np.concatenate((np.ones(int(self.bs / 2)), np.zeros(int(self.bs / 2))))
        idx_x = np.array(idx_x).flatten()
        return self.features[idx_x], torch.FloatTensor(y_batch)

class DataLoaderN:
    def __init__(self, feature, l_list, ul_list, b_size, b_size_qry, nb_task, device):
        self.feature = feature
        self.labeled_l = l_list
        self.unlabeled_l = ul_list
        self.bs = b_size
        self.bs_qry = b_size_qry
        self.nb_task = nb_task
        self.device = device
    def getBatch(self, qry):
        # idx_l = []
        feature_l = []
        label_l = []
        feature_l_qry = []
        label_l_qry = []
        for i in range(self.nb_task):

            idx_t = np.random.choice(self.labeled_l[i], size=int(self.bs / 2), replace=False).tolist()
            idx_t += np.random.choice(self.unlabeled_l[i], size=int(self.bs / 2), replace=False).tolist()
            label_t = np.concatenate((np.ones(int(self.bs / 2)), np.zeros(int(self.bs / 2))))
            feature_l.append(self.feature[i][idx_t].to(self.device))
            label_l.append(torch.FloatTensor(label_t).to(self.device))
            if qry:
                idx_t_qry = np.random.choice(self.labeled_l[i], size=int(self.bs_qry / 2), replace=False).tolist()
                idx_t_qry += np.random.choice(self.unlabeled_l[i], size=int(self.bs_qry / 2), replace=False).tolist()
                label_t_qry = np.concatenate((np.ones(int(self.bs_qry / 2)), np.zeros(int(self.bs_qry / 2))))
                feature_l_qry.append(self.feature[i][idx_t_qry].to(self.device))
                label_l_qry.append(torch.FloatTensor(label_t_qry).to(self.device))


        return feature_l, label_l, feature_l_qry, label_l_qry


def remove_values(arr1, arr2):

    res = [e for e in arr1 if e not in arr2]
    return np.array(res)


def load_yelp(file):
    data = sio.loadmat(file)
    network = data['Network'].astype(np.float)
    labels = data['Label'].flatten()
    attributes = data['Attributes'].astype(np.float)

    return network, attributes, labels

def load_data(d):
    # data = sio.loadmat("data/{}.mat".format(data_name))
    data = sio.loadmat(d)
    network = data['Network'].astype(np.float)
    labels = data['Label'].flatten()
    attributes = data['Attributes'].astype(np.float)

    return network, attributes, labels

def normalize_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalize_feature(feature):
    # Row-wise normalization of sparse feature matrix
    rowsum = np.array(feature.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(feature)
    return mx

def sp_matrix_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def SGC_process(data_name, degree, l_ratio, tr_ratio):
    adj, features, labels = load_data(data_name)
    adj = normalize_adjacency(adj)
    # split training and validation data
    idx_anomaly = np.nonzero(labels == 1)[0]
    idx_normal = np.nonzero(labels == 0)[0]
    np.random.shuffle(idx_anomaly)
    np.random.shuffle(idx_normal)
    [ano_train, ano_test] = np.array_split(idx_anomaly, [int(tr_ratio * len(idx_anomaly))])
    [nor_train, nor_test] = np.array_split(idx_normal, [int(tr_ratio * len(idx_normal))])
    idx_test = np.concatenate((ano_test, nor_test)).tolist()
    nb_ano = int(len(idx_anomaly) * l_ratio)
    # nb_ano = 10
    idx_labeled = np.random.choice(ano_train, size=nb_ano, replace=False)
    idx_unlabeled = remove_values(idx_anomaly, idx_labeled)
    idx_unlabeled = np.concatenate((nor_train, idx_unlabeled)).tolist()

    adj = sp_matrix_to_torch_sparse_tensor(adj).float()
    # features = normalize_feature(features)
    features = torch.FloatTensor(features.toarray())
    labels = torch.FloatTensor(labels)
    #compute S^K*X
    for i in range(degree):
        features = torch.spmm(adj, features)

    return features, labels, idx_labeled, idx_unlabeled, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def main():
    pass

if __name__  == '__main__':
    main()
