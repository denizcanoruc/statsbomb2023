
from typing import Any, Callable, Dict, List, Optional, Union
import pytorch_lightning as pl
from statsbomb2023.soccermap import pixel, SoccerMap, ToSoccerMapTensor
from statsbomb2023.encoder import ConvEncoder, Decoder, Predictor, CombinedEncoder, ToSoccerEncoderTensor
from statsbomb2023.gssoccermap import EmbeddingLayer, ToGameStateSoccerMapTensor, MixingLayer
from sklearn.metrics import brier_score_loss, log_loss
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from statsbomb2023.common.config import logger as log
from statsbomb2023.common.datasets import PassesDataset
from torch.utils.data import DataLoader, Subset, random_split
from rich.progress import track
from collections import defaultdict


import time



soccermap_flt = {"features" : {
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "speed": ["speedx_a02", "speedy_a02"],
                "freeze_frame_360": ["freeze_frame_360_a0"]
                },
                "label" : ["success"],  # just a dummy lalel
                "transform" : ToSoccerMapTensor(dim=(68, 104))}

soccerencoder_flt = {"features" : {
                "endlocation": ["end_x_a0", "end_y_a0"],
                "psm": ["psm"],
                "pass_team_id" : ["possession_team_id_a0"],
                "remain_min" : ["time_seconds_overall_a0"],
                "goalscore" : ["goalscore_diff"],
                "home_away" : ["is_home_a0"],
                },
                "label" : ["success"],  # just a dummy lalel
                "transform" : ToSoccerEncoderTensor()}

gssoccermap_flt = {"features" : {
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "speed": ["speedx_a02", "speedy_a02"],
                "freeze_frame_360": ["freeze_frame_360_a0"],
                "pass_team_id" : ["possession_team_id_a0"],
                "remain_min" : ["time_seconds_overall_a0"],
                "goalscore" : ["goalscore_diff"],
                "home_away" : ["is_home_a0"],
                },
                "label" : ["success"],  # just a dummy lalel
                "transform" : ToGameStateSoccerMapTensor(dim=(68, 104))}


class SoccerEncoderModule(pl.LightningModule):
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()
        

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.conven = ConvEncoder()
        self.comen = CombinedEncoder()
        self.decoder = Decoder()
        self.predictor = Predictor()

        self.loss1 = torch.nn.BCELoss()
        self.loss2 = torch.nn.CrossEntropyLoss()
        self.loss3 = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        psm, team, ha, rt, gd  = x
        convx = self.conven(psm)
        embedding = self.comen(torch.cat((convx,team),axis=1))
        return embedding

    def step(self, batch: Any):
        psm, team, ha, rt, gd = batch
        convx = self.conven(psm)
        embedding = self.comen(torch.cat((convx,team),axis=1))
        decoded = self.decoder(embedding)
        ha_hat, rt_hat, gd_hat = self.predictor(decoded)
        loss1 = self.loss1(ha_hat,ha) * 16
        loss2 = self.loss2(rt_hat,rt) * 3
        loss3 = self.loss3(gd_hat,gd) * 4
        return (loss1, loss2, loss3) , (ha_hat, rt_hat, gd_hat), (ha, rt, gd)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        total_loss = loss[0] + loss[1] + loss[2]
        # log train metrics
        self.log_dict({"train/loss" : total_loss, "train/loss_ha" : loss[0], "train/loss_rt" : loss[1], "train/loss_gd" : loss[2]}, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": total_loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        total_loss = loss[0] + loss[1] + loss[2]
        # log train metrics
        self.log_dict({"val/loss" : total_loss, "val/loss_ha" : loss[0], "val/loss_rt" : loss[1], "val/loss_gd" : loss[2]}, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": total_loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        total_loss = loss[0] + loss[1] + loss[2]
        # log train metrics
        self.log_dict({"test/loss" : total_loss, "test/loss_ha" : loss[0], "test/loss_rt" : loss[1], "test/loss_gd" : loss[2]}, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": total_loss, "preds": preds, "targets": targets}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        print(self.hparams.lr)
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)


class SoccerMapModule(pl.LightningModule):
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels=9)
        self.softmax = nn.Softmax(2)

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        return x

    def step(self, batch: Any):
        x, mask, y = batch
        surface = self.forward(x)
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

class GameStateSoccerMapModule(pl.LightningModule):
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.embedding = EmbeddingLayer()
        self.mixing = MixingLayer()
        self.model = SoccerMap(in_channels=9)
        self.softmax = nn.Softmax(2)

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, input):
        gs, x = input
        embed = self.embedding(gs)
        embed = embed.unsqueeze(2)
        embed = embed.unsqueeze(3)
        embed = embed.repeat(1,1,x.size(2),x.size(3))
        mix = self.mixing(torch.cat((x,embed),1))
        x = self.model(mix)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        return x

    def step(self, batch: Any):
        gs, x, mask, y = batch
        surface = self.forward((gs,x))
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        gs, x, _, _ = batch
        surface = self((gs,x))
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
    

def initialize_dataset(dataset: Union[PassesDataset, Callable], flt) -> PassesDataset:
        if callable(dataset):
            return dataset(xfns=flt["features"], yfns=flt["label"], transform=flt["transform"])
        return dataset

def train_module(
        model,
        dataset,
        optimized_metric=None,
        callbacks=None,
        logger=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        **train_cfg,
    ) -> Optional[float]:
    mlflow.pytorch.autolog()

    # Init lightning trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **train_cfg["trainer"])
    # Load data
    if isinstance(model, SoccerMapModule):
        data = initialize_dataset(dataset,soccermap_flt)
    elif isinstance(model, SoccerEncoderModule):
        data = initialize_dataset(dataset,soccerencoder_flt)
    elif isinstance(model, GameStateSoccerMapModule):
        data = initialize_dataset(dataset,gssoccermap_flt)
    else:
        raise ValueError("No such module type.")
    nb_train = int(len(data) * 0.8)
    lengths = [nb_train, len(data) - nb_train]
    _data_train, _data_val = random_split(data, lengths)
    train_dataloader = DataLoader(
        _data_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        _data_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # Train the model
    log.info("Fitting model on train set")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    if optimized_metric is not None:
        return trainer.callback_metrics[optimized_metric]

    return None

def test_module(
        model, dataset, batch_size=1, num_workers=0, pin_memory=False, **test_cfg
    ) -> Dict[str, float]:
        # Load data
    if isinstance(model, SoccerMapModule):
        data = initialize_dataset(dataset,soccermap_flt)
    elif isinstance(model, SoccerEncoderModule):
        data = initialize_dataset(dataset,soccerencoder_flt)
    elif isinstance(model, GameStateSoccerMapModule):
        data = initialize_dataset(dataset,gssoccermap_flt)
    else:
        raise ValueError("No such module type.")
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Switch to test mode
    torch.set_grad_enabled(False)
    model.eval()

    # Apply model on test set
    all_preds, all_targets = [], []
    for batch in track(dataloader):
        loss, y_hat, y = model.step(batch)
        all_preds.append(y_hat)
        all_targets.append(y)
    all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
    all_targets = torch.cat(all_targets, dim=0).detach().numpy()[:, 0]

    # Compute metrics
    if isinstance(model, SoccerMapModule):
        metrics = {
            "log_loss": log_loss(all_targets, all_preds, labels=[0, 1]),
            "brier": brier_score_loss(all_targets, all_preds),
        }
    return metrics
    
def predict(
    model, dataset, game_id=None, batch_size=1, num_workers=0, pin_memory=False, **predict_cfg
) -> Dict:
    # Load dataset
    if isinstance(model, SoccerMapModule):
        data = initialize_dataset(dataset,soccermap_flt)
    elif isinstance(model, SoccerEncoderModule):
        data = initialize_dataset(dataset,soccerencoder_flt)
    elif isinstance(model, GameStateSoccerMapModule):
        data = initialize_dataset(dataset,gssoccermap_flt)
    else:
        raise ValueError("No such module type.")
    actions = data.features.reset_index()
    if game_id is not None:
        actions = actions[actions.game_id == game_id]
        data = Subset(data, actions.index.values)
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    #TODO: Chnage after test_cfg
    predictor = pl.Trainer(accelerator = "gpu", devices = 1)
    predictions = torch.cat(predictor.predict(model, dataloaders=dataloader))

    output = defaultdict(dict)
    for i, action in actions.iterrows():
        if isinstance(model, SoccerMapModule) or isinstance(model, GameStateSoccerMapModule):
            output[action.game_id][action.action_id] = predictions[i][0].detach().numpy()
        else:
            output[action.game_id][action.action_id] = predictions[i].detach().numpy()
    return dict(output)