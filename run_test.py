from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
from glob import glob
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')


from pytorch_lightning import Callback
import torch.distributed as dist
import os

from experiment_framework import LitModel, get_datasets, create_confusion_scatter

class PredictionCallback(Callback):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        predictions = outputs["predictions"]
        labels = outputs["labels"]

        if trainer.world_size > 1:
            gathered_predictions = [torch.zeros_like(predictions) for _ in range(trainer.world_size)]
            gathered_labels = [torch.zeros_like(labels) for _ in range(trainer.world_size)]

            dist.all_gather(gathered_predictions, predictions)
            dist.all_gather(gathered_labels, labels)

            if trainer.is_global_zero:
                self.predictions.extend(gathered_predictions)
                self.labels.extend(gathered_labels)
        else:
            self.predictions.append(predictions)
            self.labels.append(labels)

OUTPUT_DIR='/home/experiment_framework/outputs/2023-03-21/23-09-53'

if __name__ == "__main__":
    cfg = OmegaConf.load(f'{OUTPUT_DIR}/.hydra/config.yaml')
    cfg.data.batch_size=4
    cfg.training.debug=True

    model = instantiate(cfg.model)
    datasets = get_datasets(cfg)

    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path=glob(f'{OUTPUT_DIR}/saved_models/*')[0],
        cfg=cfg,
        model=model,
        data_train=datasets['train'],
        data_valid=datasets['valid']
    )

    model_name = cfg.model.model_name if 'model_name' in cfg.model.keys() else cfg.model._target_
    print(f'{model_name = }')

    cfg.trainer.devices = 1 
    cfg.data.num_workers = 2
    cfg.trainer.accelerator = 'cuda' 
    cfg.trainer.strategy = 'auto' 

    prediction_callback = PredictionCallback()
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[prediction_callback],
        logger=False
    )

    _ = trainer.predict(lit_model, dataloaders=lit_model.val_dataloader())

    predictions = torch.cat(prediction_callback.predictions).cpu().numpy().squeeze()
    labels = torch.cat(prediction_callback.labels).cpu().numpy()

    print(f'{predictions.shape = }')
    print(f'{labels.shape = }')

    create_confusion_scatter(
        predictions,
        labels, 
        model_name,
        cfg.data.channels[0],
        OUTPUT_DIR + '/results.png'
    )
    results = pd.DataFrame({"predictions": predictions, "labels": labels})
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    results.to_csv(csv_path, index=False)