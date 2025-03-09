import torch
import torch.nn as nn

import numpy as np

class Coach:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        # store
        self.train_epoch_loss = []
        self.test_epoch_loss = []

    def train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        train_batch_loss = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.model(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_batch_loss.append(loss.item())
        self.train_epoch_loss = np.mean(train_batch_loss)
            
    def test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        test_batch_loss = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)

                test_batch_loss.append(loss.item())
        self.test_epoch_loss = np.mean(test_batch_loss)
