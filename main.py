import logging

import lightning as pl
import torch

from hydra.utils import instantiate, get_original_cwd
import hydra

from experiment_framework import (
    LitModel, 
    get_datasets
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_GPUS=torch.cuda.device_count()
DEVICE='cuda' if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # set original working dir for file access
    cfg.data.orig_dir = get_original_cwd()
    print(f'Working directory: {cfg.data.orig_dir}')

    # overwrite trainer config based on machine
    cfg.trainer.devices = NUM_GPUS
    cfg.data.num_workers = NUM_GPUS
    cfg.trainer.accelerator = DEVICE


    model = instantiate(cfg.model)
    datasets = get_datasets(cfg)

    lit_model = LitModel( 
        cfg=cfg, 
        model=model, 
        data_train=datasets['train'],
        data_valid=datasets['valid']
    )

    cfg.tb_logger.save_dir += cfg.data.folder_path.split('/')[-1]
    tb_logger = instantiate(cfg.tb_logger)
    tb_logger.log_hyperparams(cfg.model)
    trainer = pl.Trainer(
        callbacks=[
            instantiate(cfg.callbacks.learning_rate_monitor),
            instantiate(cfg.callbacks.model_checkpoint),
        ],
        logger=[tb_logger],
        **cfg.trainer
    )
    trainer.fit(
        lit_model
    )

if __name__ == '__main__':
    main()