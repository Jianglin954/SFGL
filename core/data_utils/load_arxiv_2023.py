import ipdb
import torch
import pandas as pd
import numpy as np
import torch
import random


def get_raw_text_arxiv_2023(use_text=False, seed=0, ratio=100):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  
    random.seed(seed)  

    data = torch.load('dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])


    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])


    train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])

    # ipdb.set_trace()
    if ratio == 46198:
        remove_indices = -1
        print("Using all train_mask labels for training!")
    elif ratio == 404:
        print("No training, direct testing!")
    else:
        train_index = torch.squeeze(torch.nonzero(train_mask))
        num_elements_to_remove = train_index.shape[0] - ratio
        remove_indices = random.sample(list(train_index.cpu().numpy()), num_elements_to_remove)
        remove_indices = np.array(remove_indices)
        remove_indices = torch.from_numpy(remove_indices)   
        train_mask[remove_indices] = False  

    print(f"Labeled samping in training set: {torch.sum(train_mask)}/{train_mask.shape[0]}, ratio: {ratio}%!!")

    data.train_mask = train_mask

    print(f"Before pseudo labling, labeled samples in training set: {torch.sum(data.train_mask)}/{train_mask.shape[0]}, ratio: {ratio}%!!")


    ''' GNN training requires the following code to be commented out, GNN training uses original labeled data and does not use pseudo-labeled data
    # ipdb.set_trace()
    if ratio == 50:
        GCN_KNN_pred_y = torch.load('./pred_labels/arxiv_2023_pred_labels_seed3_ratio50_K25_acc40086580086580087.pt')
        GCN_KNN_pred_y2 = torch.load('./pred_labels/ogbn-arxiv_pred_labels_seed3_ratio50_K15_acc33901199514433267.pt')

    elif ratio == 100:
        GCN_KNN_pred_y = torch.load('./pred_labels/arxiv_2023_pred_labels_seed4_ratio100_K25_acc44545454545454544.pt')
    elif ratio == 150:
        GCN_KNN_pred_y = torch.load('./pred_labels/arxiv_2023_pred_labels_seed4_ratio150_K25_acc4458874458874459.pt')
    elif ratio == 300:
        GCN_KNN_pred_y = torch.load('./pred_labels/arxiv_2023_pred_labels_seed3_ratio300_K25_acc48874458874458876.pt')
    elif ratio == 500:
        GCN_KNN_pred_y = torch.load('./pred_labels/arxiv_2023_pred_labels_seed4_ratio500_K25_acc5112554112554113.pt')
    else:
        print("labels errors.")
    GCN_KNN_pred_y = torch.squeeze(GCN_KNN_pred_y).cpu()        # ar23: data.y = torch.Size([46198])
    untrain_mask = data.val_mask + data.test_mask   #
    all_train_mask = ~untrain_mask                  #
    supple_mask = all_train_mask.float() - data.train_mask.float()  #
    supple_mask = supple_mask > 0
    print(f"all training number: {torch.sum(all_train_mask)}, supple mask number: {torch.sum(supple_mask)}!!!")
    # ipdb.set_trace()
    data.y[supple_mask] = GCN_KNN_pred_y[supple_mask]    
    data.train_mask = all_train_mask


    print(f"After pseudo labling, labeled samples in training set: {torch.sum(data.train_mask)}/{train_mask.shape[0]}, ratio: {ratio}%!!")   
    '''



























    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
    return data, text
