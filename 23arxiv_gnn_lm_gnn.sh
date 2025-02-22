

:<<BLOCK
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 50  | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 100 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 150 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 300 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 500 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt

CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 50  | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 100 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 150 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 300 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 500 | tee -a ./tmp_results/GL/ar23_gnn_lm_gnn.txt
BLOCK


CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 50  | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 100 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 150 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 300 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type TA dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 500 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt

CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 50  | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 100 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 150 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 300 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
CUDA_VISIBLE_DEVICES=1 python -m core.trainGNN_lmf gnn.train.feature_type E dataset arxiv_2023 gnn.train.K 25 gnn.train.lr 0.01 ratio 500 | tee -a ./tmp_results/GL/ar23_gnn_ori_lm_gnn.txt
