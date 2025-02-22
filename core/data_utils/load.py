import os
import json

# import ipdb
import torch
import csv
from core.data_utils.dataset import CustomDGLDataset
import ipdb
from core.utils import time_logger, top_k, constGraph, constLargeGraph, constLargeGraph23, constGraph_withValues, nearest_neighbors, normalize


def load_gpt_preds(dataset, topk):
    preds = []
    with open(f'gpt_preds/{dataset}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0, ratio=100):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'arxiv_2023':
        from core.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed, ratio=ratio)
        if use_dgl:     # False
            from torch_geometric.utils import dense_to_sparse
            Adj = torch.from_numpy(nearest_neighbors(data.x.numpy(), 10, 'cosine'))
            Adj = normalize(Adj, 'sym', sparse=False)
            edge_index, edge_attr = dense_to_sparse(Adj)
            data.edge_index = edge_index
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed, ratio=ratio)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed, ratio=ratio)

    return data, num_classes, text
