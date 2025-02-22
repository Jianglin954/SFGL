





#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  0.001 lm.train.batch_size 20 ratio 50 |  tee -a ./tmp_results/LM/ar23_LMs_pls.txt
#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  0.001 lm.train.batch_size 20 ratio 100 |  tee -a ./tmp_results/LM/ar23_LMs_pls.txt
#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  0.001 lm.train.batch_size 20 ratio 150 |  tee -a ./tmp_results/LM/ar23_LMs_pls.txt
#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  0.001 lm.train.batch_size 20 ratio 300 |  tee -a ./tmp_results/LM/ar23_LMs_pls.txt
#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  0.001 lm.train.batch_size 20 ratio 500 |  tee -a ./tmp_results/LM/ar23_LMs_pls.txt






#WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  2e-05 lm.train.batch_size 20 ratio 50 |  tee -a ./tmp_results/LM/ar23_LMs_pls2e05.txt
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  2e-05 lm.train.batch_size 20 ratio 100 |  tee -a ./tmp_results/LM/ar23_LMs_pls2e05.txt
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  2e-05 lm.train.batch_size 20 ratio 150 |  tee -a ./tmp_results/LM/ar23_LMs_pls2e05.txt
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  2e-05 lm.train.batch_size 20 ratio 300 |  tee -a ./tmp_results/LM/ar23_LMs_pls2e05.txt
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM_pls dataset arxiv_2023 lm.train.lr  2e-05 lm.train.batch_size 20 ratio 500 |  tee -a ./tmp_results/LM/ar23_LMs_pls2e05.txt

