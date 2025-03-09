import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.utils.seed import fix_seed
from src.data.mnist_data import MNISTData
from src.models.nn_model import NNModel
from src.models.coach import Coach
from src.models.evaluate import evaluate
from src.visualization.visualize import plot

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 3
CLASSES = 10

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_module = MNISTData(batch_size=BATCH_SIZE)
    train_loader, test_loader = data_module.get_dataloaders()

    model = NNModel(classes=CLASSES, dropout_prob=.5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    train_loss, test_loss = [], []
    coach = Coach(model, train_loader, test_loader, loss_fn, optimizer, device)
    for epoch in range(EPOCHS):
        coach.train_epoch()
        coach.test_epoch()
        
        print("epoch: ", epoch+1, "/", EPOCHS)
        train_epoch_loss, test_epoch_loss = coach.train_epoch_loss, coach.test_epoch_loss
        print("train loss: ", train_epoch_loss)
        print("test loss: ", test_epoch_loss)
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)

    plot(train_loss, test_loss)

    trues, preds = evaluate(model, test_loader, device)
    accuracy = accuracy_score(trues, preds)

    print("accuracy: ", accuracy)

    torch.save(model.state_dict(), "model")

if __name__ == "__main__":
    fix_seed()
    main()