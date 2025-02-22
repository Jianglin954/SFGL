import os
import numpy as np
import time
import datetime
import pytz
import torch
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor
import ipdb
from sklearn.neighbors import kneighbors_graph

EOS = 1e-10

def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

def top_k(raw_graph, K):
    # ipdb.set_trace()
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def top_k_withValues(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    # ipdb.set_trace()
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask

    _index = torch.zeros(2, raw_graph.shape[0] * (int(K))).cuda()
    _index[0, :] = torch.arange(0, raw_graph.shape[0]).view(-1, 1).repeat(1, int(K)).view(-1)
    _index[1, :] = indices.view(-1)
    _index = _index.long()

    _val = values.view(-1)
    return sparse_graph, _val, _index

def constGraph(features, k):
    print("Constructing a graph ...")
    print("No F.normalize ...")
    # features = F.normalize(features, dim=1, p=2)
    similarities = torch.mm(features, features.t())
    # ipdb.set_trace()
    similarities = top_k(similarities, k + 1)
    index = torch.nonzero(similarities).t()
    print("Finished graph construction!...")
    return index

def constGraph_withValues(features, k):
    print("Constructing a graph ...")
    print("No F.normalize ...")
    # features = F.normalize(features, dim=1, p=2)
    similarities = torch.mm(features, features.t())

    inv_sqrt_degree = 1. / (torch.sqrt(similarities.sum(dim=1, keepdim=False)) + 1e-10)
    similarities = inv_sqrt_degree[:, None] * similarities * inv_sqrt_degree[None, :]

    similarities, values, _index = top_k_withValues(similarities, k + 1)
    print("Finished graph construction!...")
    return _index, values

def constLargeGraph(features, k, step=5000):
    features = F.normalize(features, dim=1, p=2)    # # 在arxiv上: 使用ogb的特征时，需要使用F.normalize; 使用Prediction特征时，不需要使用 F.normalize
    print("Constructing a large graph ...")
    idx = 0
    _index = torch.zeros(2, features.shape[0] * (k + 1)).cuda() # torch.Size([2, 2709488]) 2709488=169343*16

    while idx < features.shape[0]:
        if (idx + step) > features.shape[0]:
            end = features.shape[0]
        else:
            end = idx + step
        sub_features = features[idx:(idx + step)]
        sub_similarities = torch.mm(sub_features, features.t())
        values, indices = sub_similarities.topk(k=k + 1, dim=-1)
        cols = indices.view(-1)
        rows = torch.arange(idx, end).view(-1, 1).repeat(1, k + 1).view(-1)
        rows = rows.long()
        cols = cols.long()
        _index[0, (idx * (k + 1)):(end * (k + 1))] = rows
        _index[1, (idx * (k + 1)):(end * (k + 1))] = cols
        idx += step
    # ipdb.set_trace()
    _index = _index.long()
    edge_index = SparseTensor(
        row=_index[0],
        col=_index[1],
        # value=value,
        # sparse_sizes=size,
        is_sorted=True,
        trust_data=True,
    )
    del _index
    print("Finished large graph construction!...")
    return edge_index




def constLargeGraph23(features, k, step=5000):
    features = F.normalize(features, dim=1, p=2)
    print("Constructing a large graph ...")
    idx = 0
    _index = torch.zeros(2, features.shape[0] * (k + 1)).cuda()

    while idx < features.shape[0]:
        if (idx + step) > features.shape[0]:
            end = features.shape[0]
        else:
            end = idx + step
        sub_features = features[idx:(idx + step)]
        sub_similarities = torch.mm(sub_features, features.t())
        values, indices = sub_similarities.topk(k=k + 1, dim=-1)
        cols = indices.view(-1)
        rows = torch.arange(idx, end).view(-1, 1).repeat(1, k + 1).view(-1)
        rows = rows.long()
        cols = cols.long()
        _index[0, (idx * (k + 1)):(end * (k + 1))] = rows
        _index[1, (idx * (k + 1)):(end * (k + 1))] = cols
        idx += step
    # ipdb.set_trace()
    _index = _index.long()
    print("Finished large graph construction!...")
    return _index


'''
def nearest_neighbors(X, k, metric, p):
    adj = kneighbors_graph(X, k, metric=metric, p=p)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj

'''
def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

