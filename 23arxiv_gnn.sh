
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 50    gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 100   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 150   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 300   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 500   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 1000  gnn.train.lr 0.01 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt


#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 50    gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 100   gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 150   gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 300   gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 500   gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 15 ratio 1000  gnn.train.lr 0.1 | tee -a ./tmp_results/GL/arxiv23_#_train_GNN.txt


#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 50    gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 100   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 150   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 300   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 500   gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt
#CUDA_VISIBLE_DEVICES=0 python -m core.trainGNN gnn.train.feature_type ogb dataset arxiv_2023 gnn.train.K 25 ratio 1000  gnn.train.lr 0.01 | tee -a ./tmp_results/GL/ar23_train_GNN.txt

