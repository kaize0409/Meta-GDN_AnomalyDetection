import argparse
import numpy as np
import torch
from time import perf_counter
import networkx as nx
import random
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
# from data import largest_connected_components
from datetime import datetime
import scipy.io as sio
from collections import OrderedDict

def aucPerformance(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    ap = average_precision_score(y_true, y_pred)
    print("AUC-ROC: %.4f, AUC-PR: %.4f, AP: %.4f" % (roc_auc, auc_pr, ap))
    return roc_auc, auc_pr, ap

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random Seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-6, help='weight decay')
    parser.add_argument('--degree', type=int, default=2, help='K in SGC')
    args, _ = parser.parse_known_args()

    return args

def sgc_precompute(features, adj, degree):
    # compute S^K
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def process_graph(adj):
    pass

def acc():
    pass

def starting_node(nd_view):
    node_c = []
    for t in nd_view:
        if t[1] <= 3:
            node_c.append(t[0])
    if len(node_c) == 0:
        return random.sample(nd_view, 1)[0][0]

    return random.sample(node_c, 1)[0]

def partition_graph(adj, nb_graphs, nb_nodes):
    # print(adj.shape[0])
    G = nx.Graph(adj)
    print("graph size:", len(G))
    nodes_list = []
    while len(G.nodes) > 3500:

        nd_view = list(G.degree)
        # print(nd_view)
        nodes_selected = []
        nodes_selected.append(starting_node(nd_view))
        for n in nodes_selected:
            nei_new = [i for i in list(G[n]) if i not in nodes_selected]
            nodes_selected.extend(nei_new)
            if len(nodes_selected) > nb_nodes:
                break

        size_s = len(nodes_selected)
        nodes_list.append(nodes_selected)
        G.remove_nodes_from(nodes_selected)

    # remaining nodes
    nodes_list.append(list(G.nodes))
    res = []
    for i in nodes_list:
        if len(i) > 3500:
            res.append(i)
    print('final size: ', len(res))
    return res