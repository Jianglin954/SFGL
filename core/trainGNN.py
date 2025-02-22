import ipdb

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
import pandas as pd
from core.config import cfg, update_cfg

import random
import numpy as np
import torch
import dgl
import os

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    # ipdb.set_trace()
    TRAINER = DGLGNNTrainer if cfg.gnn.train.use_dgl else GNNTrainer
    all_acc = []
    best_acc = 0

    tmp_file = ""
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.train()
        logits, acc = trainer.eval_and_save()
        all_acc.append(acc)
        # ipdb.set_trace()
        test_acc = acc['test_acc']

        if test_acc > best_acc:
            best_acc = test_acc
            best_logits = logits
            acc_str = str(test_acc).split('.')
            acc_str = acc_str[1]
            if tmp_file == "":
                file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_randomgraph.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_FINETUNEDBERT.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_SAGE2.pt"       # SAGE GAT
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_textencode.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_euclidean.pt" # euclidean manhattan
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_iterative_GPT_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}.pt"
                tmp_file = file_name
            else:
                os.remove(tmp_file)
                file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_randomgraph.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_FINETUNEDBERT.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_SAGE2.pt"       # SAGE GAT
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_textencode.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}_euclidean.pt"  # euclidean manhattan
                # file_name = f"./pred_labels/{cfg.dataset}_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}.pt"
                # file_name = f"./pred_labels/{cfg.dataset}_iterative_GPT_pred_labels_seed{cfg.seed}_ratio{cfg.ratio}_K{cfg.gnn.train.K}_acc{acc_str}.pt"
                tmp_file = file_name
            torch.save(logits.argmax(dim=-1, keepdim=True), file_name)
        ### save pseudo labels generated from gnn with KNN graph
        ## logits.argmax(dim=-1, keepdim=True).shape
        ## torch.save(logits.argmax(dim=-1, keepdim=True), './pred_labels/arxiv_pred_labels_seed0_acc7344.pt')
    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")


if __name__ == '__main__':

    cfg = update_cfg(cfg)
    run(cfg)
    print(cfg)