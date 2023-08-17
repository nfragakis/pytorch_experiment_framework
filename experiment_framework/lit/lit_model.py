import torch
import lightning as pl
from torch.utils import data 
import torch.nn.functional as F

from hydra.utils import instantiate
from omegaconf import DictConfig 

class LitModel(pl.LightningModule):
    """
    LitModel is a PyTorch Lightning Module that encapsulates a given model, 
    training and validation data, and a loss function. It also provides methods 
    for data loading, optimizer configuration, and steps for training, validation, 
    and testing.
    """
    def __init__(self, cfg, model, data_train, data_valid):
        """
        Initializes the LitModel with the given configuration, model, training data, 
        and validation data. It also instantiates the loss function from the configuration.
        """
        super(LitModel, self).__init__()
        self.cfg = cfg
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.loss_fn = instantiate(cfg.loss)

    def forward(self, x, *args, **kwargs):
        """
        Defines the forward pass for the encapsulated model.
        """
        return self.model(x)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training data.
        """
        return data.DataLoader(
            self.data_train,
            sampler=sampler,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation data.
        """
        return data.DataLoader(
            self.data_valid,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        """
        Returns a DataLoader for the validation data for prediction.
        """
        return data.DataLoader(
            self.data_valid,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False
        )

    def configure_optimizers(self):
        """
        Configures and returns the optimizer and scheduler based on the given configuration.
        """
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.cfg.scheduler.obj, optimizer=optimizer)

        return [optimizer], [{"scheduler": scheduler, "interval": self.cfg.scheduler.interval}]

    def training_step(self, batch, batch_idx):
        """
        Defines the training step. It runs the batch through the model, logs the loss, 
        and returns the outputs.
        """
        x, y, preds, loss = self._run_on_batch(batch)

        self.log("train/loss", loss)
        outputs = {"loss": loss}
        self.add_on_first_batch({"preds": preds.detach()}, outputs, batch_idx)

        return outputs

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step. It runs the batch through the model, logs the loss, 
        and returns the outputs.
        """
        x, y, preds, loss = self._run_on_batch(batch)
        self.log("validation/loss", loss)
        outputs = {"loss": loss}
        self.add_on_first_batch({"preds": preds.detach()}, outputs, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        """
        Defines the test step. It runs the batch through the model and logs the loss.
        """
        x, y, preds, loss = self._run_on_batch(batch)
        self.log("test/loss", loss)

    def predict_step(self, batch, batch_idx):
        """
        Defines the prediction step. It runs the batch through the model and returns 
        the predictions and labels.
        """
        x, y = batch
        y_pred = self(x)
        output = {"predictions": y_pred, "labels": y}
        return output

    def _run_on_batch(self, batch, with_preds=False):
        """
        Helper method to run a batch through the model and calculate the loss.
        """
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds.squeeze(), y.squeeze())

        return x, y, preds, torch.Tensor(loss)

    def add_on_first_batch(self, metrics, outputs, batch_idx):
        """
        Helper method to add metrics to the outputs if it's the first batch.
        """
        if batch_idx == 0:
            outputs.update(metrics)

    def add_on_logged_batches(self, metrics, outputs):
        """
        Helper method to add metrics to the outputs if it's a logged batch.
        """
        if self.is_logged_batch:
            outputs.update(metrics)

    def is_logged_batch(self):
        """
        Helper method to check if the current batch should be logged.
        """
        if self.trainer is None:
            return False
        else:
            return self.trainer._logger_connector.should_update_logs