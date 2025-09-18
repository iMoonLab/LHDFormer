from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import random
from .utils.device import device_set
device = device_set()
import numpy as np
import torch
from .utils import Logger_main


def model_training(cfg: DictConfig):
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)
    t_acc, t_auc, t_sen, t_spec = training.train()
    return t_acc, t_auc, t_sen, t_spec


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    acc_list = []
    auc_list = []
    sen_list = []
    spec_list = []
    seeds = list(range(cfg.repeat_time))
    logger = Logger_main()

    for it in range(cfg.repeat_time):
        SEED = seeds[it] + 1
        logger.info(f"Fold {it + 1}/{len(seeds)}, SEED: {SEED}, device:{device}")
        random.seed(SEED)  # set the random seed so that the random permutations can be reproduced again
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        run = wandb.init(project=cfg.project, reinit=True,
                         group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        t_acc, t_auc, t_sen, t_spec = model_training(cfg)
        acc_list.append(t_acc)
        auc_list.append(t_auc)
        sen_list.append(t_sen)
        spec_list.append(t_spec)
        run.finish()
    print(device)
    print("test acc mean:{}  std: {}".format(np.mean(acc_list), np.std(acc_list)))
    print("test auc mean:{}  std: {}".format(np.mean(auc_list), np.std(auc_list)))
    print("test sensitivity mean:{}  std: {}".format(np.mean(sen_list), np.std(sen_list)))
    print("test specficity mean:{}  std: {}".format(np.mean(spec_list), np.std(spec_list)))


if __name__ == '__main__':
    main()
