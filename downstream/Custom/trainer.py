import argparse
import os
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

from .dataloader import CustomEmoDataset
from utils.metrics import ConfusionMetrics

from pretrain.trainer import PretrainedRNNHead
from tqdm import tqdm

class DownstreamGeneral(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hp = hparams
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)

        if self.hp.pretrained_path is not None:
            self.model = PretrainedRNNHead.load_from_checkpoint(self.hp.pretrained_path, strict=False,
                                                                n_classes=self.dataset.nemos,
                                                                backend=self.hp.model_type)
        else:
            self.model = PretrainedRNNHead(n_classes=self.dataset.nemos,
                                           backend=self.hp.model_type)
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor(
            [counter[c] for c in self.dataset.emoset]
        ).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(
            f"Weigh losses by prior distribution of each class: {weights}."
        )

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Define metrics
        if hasattr(self.dataset, 'val_dataset'):
            self.valid_met = ConfusionMetrics(self.dataset.nemos)
        if hasattr(self.dataset, 'test_dataset'):
            self.test_met = ConfusionMetrics(self.dataset.nemos)

    def forward(self, x, length):
        return self.model(x, length)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.trainable_params(), lr=self.hp.lr)
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(dataset=self.dataset.train_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=True,
                            num_workers=self.hp.nworkers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if not hasattr(self.dataset, 'val_dataset'):
            return
        loader = DataLoader(dataset=self.dataset.val_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=False,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(dataset=self.dataset.test_dataset,
                            batch_size=1,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def training_step(self, batch, batch_idx):
        feats, length, label = batch
        pout = self(feats, length)
        loss = self.criterion(pout, label)
        tqdm_dict = {'loss': loss}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label = batch
        pout = self(feats, length)
        loss = self.criterion(pout, label)
        for l, p in zip(label, pout):
            self.valid_met.fit(int(l), int(p.argmax()))
        self.log('valid_loss', loss, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        print (self.valid_met.uar)
        self.log('valid_UAR', self.valid_met.uar)
        self.log('valid_WAR', self.valid_met.war)
        self.log('valid_macroF1', self.valid_met.macroF1)
        self.log('valid_microF1', self.valid_met.microF1)
        self.valid_met.clear()

    def test_step(self, batch, batch_idx):
        feats, label = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        pout = self(feats, length)
        self.test_met.fit(int(label), int(pout.argmax()))

    def on_test_epoch_end(self):
        """Report metrics."""
        self.log('test_UAR', self.test_met.uar, logger=True)
        self.log('test_WAR', self.test_met.war, logger=True)
        self.log('test_macroF1', self.test_met.macroF1, logger=True)
        self.log('test_microF1', self.test_met.microF1, logger=True)

        print(f"""++++ Classification Metrics ++++
                  UAR: {self.test_met.uar:.4f}
                  WAR: {self.test_met.war:.4f}
                  macroF1: {self.test_met.macroF1:.4f}
                  microF1: {self.test_met.microF1:.4f}""")
