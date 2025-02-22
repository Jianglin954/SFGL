import torch
from time import time
import numpy as np

from core.GNNs.GCN.model import GCN
from core.GNNs.SAGE.model import SAGE
from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger, top_k, constGraph, constLargeGraph, constLargeGraph23, constGraph_withValues, nearest_neighbors, normalize
import ipdb
LOG_FREQ = 10



class GNNTrainer():

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs
        self.k = cfg.gnn.train.K
        self.ratio = cfg.ratio
        # ! Load data

        data, num_classes = load_data(self.dataset_name, use_dgl=False,
                         use_text=False, seed=self.seed, ratio=self.ratio)

        self.num_nodes = data.x.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5

        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            # ipdb.set_trace()
            features = data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}_GSL_data_split_no_supple_mask/{self.lm_model_name}-seed{self.seed}.emb"
            # LM_emb_path = f"prt_lm/{self.dataset_name}_#_train_LMs_pls/{self.lm_model_name}-seed{self.seed}-labels{self.ratio}-lr{2e-05}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.emb"
            # LM_emb_path = f"prt_lm/{self.dataset_name}2_#_train_LMs_pls/{self.lm_model_name}-seed{self.seed}-labels{self.ratio}-lr{2e-05}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        elif self.feature_type == 'TEXTENCODER':
            print("Loading TEXTENCODER  features ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}_#_train_LMs_pls/{self.lm_model_name}-seed{self.seed}-labels{self.ratio}-lr{2e-05}-textencode.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
            print("Loaded TEXTENCODER  features ...")
        elif self.feature_type == 'FINETUNEDBERT':
            print("Loading TEXTENCODER  features ...")
            LM_emb_path = f"prt_lm/{self.dataset_name}_#_train_LMs_pls/{self.lm_model_name}-seed{self.seed}-labels{self.ratio}-lr{2e-05}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
            print("Loaded TEXTENCODER  features ...")
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x



        # ipdb.set_trace()

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        if self.dataset_name == 'cora' or self.dataset_name == 'pubmed':       #
            # edge_index = constGraph(self.features, self.k)
            # edge_index, values = constGraph_withValues(self.features, self.k)
            # self.data.edge_index = edge_index
            # self.data.edge_weight = values
            # ipdb.set_trace()

            Adj = torch.from_numpy(nearest_neighbors(features.numpy(), self.k, 'cosine')).cuda()
            ## Adj = torch.from_numpy(nearest_neighbors(features.numpy(), self.k, metric="euclidean")).cuda()  # euclidean manhattan

            if self.gnn_model_name != "SAGE":
                print("using Graph Normalization!")
                Adj = normalize(Adj, 'sym', sparse=False)

            nonzero_indices = torch.where(Adj != 0)
            nonzero_values = Adj[nonzero_indices]
            edge_index = torch.zeros(2, nonzero_indices[0].shape[0]).cuda()
            edge_index[0, :] = nonzero_indices[0]
            edge_index[1, :] = nonzero_indices[1]
            edge_index = edge_index.long()

            self.data.edge_index = []
            self.data.edge_index = edge_index
            self.data.edge_weight = nonzero_values


            # pass
        elif self.dataset_name == 'pubmed2':
            pass
        elif self.dataset_name == 'ogbn-arxiv':
            edge_index = constLargeGraph(self.features, self.k, step=1000)
            self.data.edge_index = []
            self.data.edge_index = edge_index
            # pass
        elif self.dataset_name == 'arxiv_2023':
            # edge_index = constLargeGraph23(self.features, self.k, step=1000)

            def generate_random_graph_with_self_loops_undirected(n, p):
                """
                Generate a random undirected graph G(n, p) with self-connections and return its edge index.
            
                Parameters:
                    n (int): Number of nodes in the graph.
                    p (float): Probability of edge creation between nodes (0 <= p <= 1).
            
                Returns:
                    edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges), including self-connections.
                """
                # Generate all possible pairs of nodes (i, j) including self-loops
                row, col = torch.triu_indices(n, n, offset=0)  # Include diagonal for self-loops
            
                # Randomly decide which edges to include based on probability p
                mask = torch.rand(row.size(0)) < p
                row, col = row[mask], col[mask]
            
                # Combine into an edge index (directed upper triangular)
                edge_index = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
            
                # Add reverse edges to make the graph undirected
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
                return edge_index
            ipdb.set_trace()

            edge_index = generate_random_graph_with_self_loops_undirected(self.features.shape[0], p=0.01)
            edge_index = edge_index.to(self.device)

            self.data.edge_index = []
            self.data.edge_index = edge_index
            # pass






        # ! Trainer init
        use_pred = self.feature_type == 'P'
        if self.gnn_model_name == "GCN":
            self.model = GCN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                             hidden_channels=self.hidden_dim,
                             out_channels=self.num_classes,
                             num_layers=self.num_layers,
                             dropout=self.dropout,
                             use_pred=use_pred).to(self.device)

        elif self.gnn_model_name == "SAGE":
            self.model = SAGE(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                              hidden_channels=self.hidden_dim,
                              out_channels=self.num_classes,
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _forward_edgeValue(self, x, edge_index, edge_weight):
        logits = self.model(x, edge_index, edge_weight)  # small-graph
        return logits



    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        # ipdb.set_trace()
        logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(
            f'[{self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
