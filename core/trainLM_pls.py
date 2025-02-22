from core.LMs.lm_trainer import LMTrainer
from core.LMs.lm_trainer_pseudo_labels import LM_PL_Trainer, LM_Text_Encoder_Trainer
from core.config import cfg, update_cfg
import pandas as pd
import ipdb

def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        # cfg.ratio = 9
        # ipdb.set_trace()
        trainer = LM_PL_Trainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
    print(cfg)
