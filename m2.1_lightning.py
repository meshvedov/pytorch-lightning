#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd

import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.utils.data as data


import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from lightning import (
    Trainer,
    LightningModule,
    LightningDataModule,    
)

from dataclasses import dataclass

torch.set_float32_matmul_precision('high')

@dataclass
class CFG:
    csv_train_path: str = "data/sign_mnist_train.csv"
    csv_test_path: str = "data/sign_mnist_test.csv"
    path_to_save: str = "models"
    seed: int = 2024
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    hidden_size: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 200
    num_workers: int = 2
    epochs: int = 5
    stride: int = 1
    dilation: int = 1
    n_classes: int = 25
    log_every_n_steps: int = 1

class SignLanguageDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        label = self.df.iloc[index, 0]

        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class SignDM(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.datasets = {}
        self.transforms4train = transforms.Compose(
            [
                # transforms.Normalize(159, 40),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
            ]
        )
        
    def prepare_data(self):
        if not os.path.exists(self.cfg.csv_train_path):
            raise FileNotFoundError(f"Training data file {self.cfg.csv_train_path} not found.")
        if not os.path.exists(self.cfg.csv_test_path):
            raise FileNotFoundError(f"Test data file {self.cfg.csv_test_path} not found.")
        
        self.data = pd.read_csv(self.cfg.csv_train_path)
        self.test = pd.read_csv(self.cfg.csv_test_path)


    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            train, val = train_test_split(self.data, test_size=0.2, random_state=self.cfg.seed)
            self.train = SignLanguageDataset(train, transform=self.transforms4train)
            self.val = SignLanguageDataset(val)
            
        if stage == 'test' or stage is None:
            self.test = SignLanguageDataset(self.test)

    def _make_dataloader(self, dataset, cfg):
        return data.DataLoader(
            dataset,
            batch_size= cfg.batch_size,
            num_workers= cfg.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train, self.cfg)

    def val_dataloader(self):
        return self._make_dataloader(self.val, self.cfg)
    
    def test_dataloader(self):
        return self._make_dataloader(self.test, self.cfg)

    def teardown(self, stage: str):
        if stage == 'fit' or stage is None:
            del self.train, self.val
        if stage == 'test' or stage is None:
            del self.test
        
class SignModel(LightningModule):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.lr
        self.stride = cfg.stride
        self.dilation = cfg.dilation
        self.n_classes = cfg.n_classes
        self.num_correct = 0
        self.num_total = 0

        self.block1 = nn.Sequential(
            # (bacth, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(8),
            #(batch, 8, 28, 28)
            nn.AvgPool2d(2),
            #(batch, 8, 14, 14)
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(16),
            #(batch, 16, 14,14)
            nn.AvgPool2d(2),
            #(batch, 16, 7, 7)
            nn.ReLU(),
        )
        self.lin1 = nn.Linear(in_features=16*7*7, out_features=100)
        #(batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=.3)
        self.lin2 = nn.Linear(in_features=100, out_features=self.n_classes)
        #(batch, 25)
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x

    def basic_step(self, batch, batch_idx, step: str):
        data, labels = batch
        pred_clas = self(data)
        loss = self.classification_criterion(pred_clas, labels)
        pred_labels = torch.argmax(pred_clas, dim=1)
        self.num_correct += float((pred_labels == labels).sum())
        self.num_total += labels.shape[0]
        loss_dict = {
            f"{step}/loss": loss,
        }
        self.log_dict(loss_dict, prog_bar=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, 'train')
        return loss['train/loss']

    def validation_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, 'valid')
        return loss['valid/loss']

    def test_step(self, batch, batch_idx):
        loss = self.basic_step(batch, batch_idx, 'test')
        return loss['test/loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _accuracy(self, stage: str):
        accuracy = self.num_correct / self.num_total
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)

    def _reset_num(self):
        self.num_correct = 0
        self.num_total = 0
        
    def on_train_epoch_start(self):
        self._reset_num()

    def on_test_epoch_start(self):
        self._reset_num()
        
    def on_test_epoch_end(self):
        self._accuracy('test')
        
    def on_validation_epoch_start(self):
        self._reset_num()
        
    def on_validation_epoch_end(self):
        self._accuracy('valid')


def main(fast_dev_run: bool, epochs: int):
    cfg = CFG()
    cfg.epochs = epochs
    ds = SignDM(cfg)
    model = SignModel(cfg)
    try:
        if fast_dev_run:
            trainer = Trainer(fast_dev_run=fast_dev_run)
            trainer.fit(model, datamodule=ds)
            print("Тестовый прогон завершился успешно")
        
        trainer = Trainer(
            accelerator=cfg.device,
            max_epochs=cfg.epochs,
            log_every_n_steps=cfg.log_every_n_steps,)
        trainer.fit(model, datamodule=ds)
        trainer.test(model, datamodule=ds)
        print("Обучение завершено успешно")
                
    except Exception as e:
        print(f"!!!EXCEPTION: {e}")
        print("!!!Прогон завершился с ошибкой!!!")
        sys.exit(1)
        
    # Сохранение весов модели
    model_path = Path(cfg.path_to_save)
    model_name = "sign_model_weight.pth"
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / model_name)
    print(f"Веса модели сохранены в {model_path / model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Lightning script")
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='Run a single batch for quick debugging')
    parser.add_argument('--epochs', type=int, default=5, help='Run a single batch for quick debugging')
    args = parser.parse_args()
    main(args.fast_dev_run, args.epochs)