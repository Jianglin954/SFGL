import ipdb

from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.gnn_trainer_lmf import GNNTrainer_lmf
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
import pandas as pd
from core.config import cfg, update_cfg

import random
import numpy as np
import torch
import dgl
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    TRAINER = DGLGNNTrainer if cfg.gnn.train.use_dgl else GNNTrainer_lmf
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.train()
        logits, acc = trainer.eval_and_save()
        all_acc.append(acc)
    # ipdb.set_trace()
    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")


if __name__ == '__main__':

    cfg = update_cfg(cfg)
    run(cfg)
    print(cfg)
