import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.utils.seed import fix_seed
from src.data.mnist_dataset import create_dataset
from src.models.nn_model import NNModel
from src.models.coach import Coach
from src.models.evaluate import evaluate
from src.visualization.visualize import plot

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    TRAIN_PATH = config_file['config']['train_path']
    TEST_PATH = config_file['config']['test_path']
    CLASSES = config_file['config']['classes']
    EPOCHS = config_file['config']['epochs']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']
    DROPOUT_PROB = config_file['config']['dropout_prob']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = create_dataset(TRAIN_PATH, TEST_PATH, BATCH_SIZE)

    model = NNModel(classes=CLASSES, dropout_prob=DROPOUT_PROB).to(device)
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