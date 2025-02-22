# Scale-Free Graph-Language Models (ICLR 2025)
Codes for paper [Scale-Free Graph-Language Models](https://openreview.net/forum?id=nFcgay1Yo9)


<img src="./sfgl.jpg">


## Contributions

1. We identify two key challenges in existing GLMs: *artificial structural assumptions in graph generation* and *unreliable LM finetuning for text embedding*. We propose addressing these challenges simultaneously by exploring a well-grounded graph structural prior.

2. We leverage the *scale-free edge distribution* in real networks as our graph structural prior. Our empirical validation and analysis reveal that a KNN graph, constructed using cosine similarity with an appropriately chosen k, effectively approximates a scale-free network.

3. To the best of our knowledge, the proposed SFGL is the first work to *unify graph generation and text embedding within a GLM framework*, highlighting the synergistic potential of GNNs and LMs under a scale-free structural prior.


## Datasets

Datasets can be download from  [here](https://github.com/XiaoxinHe/TAPE). Please place the downloaded files in the folder `dataset`.


## Installation
```bash
conda env create -f SFGL_environment.yml
```


## Usage

see bash files:
```bash
bash 23arxiv_gnn.sh
bash 23arxiv_gnn_lm.sh
bash 23arxiv_gnn_lm_gpt.sh
bash 23arxiv_gnn_lm_gnn.sh
```

The experimental results will be saved in "tmp_results" folder.

# Reference

    @inproceedings{
      lu2025scalefree,
      title={Scale-Free Graph-Language Models},
      author={Jianglin Lu and Yixuan Liu and Yitian Zhang and Yun Fu},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=nFcgay1Yo9}
    }

# Acknowledgement
Our code is mainly built on [TAPE](https://github.com/XiaoxinHe/TAPE). We sincerely appreciate the authors for their valuable contributions. 
