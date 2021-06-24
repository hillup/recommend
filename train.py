import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from data.dataset import build_dataset
from model.EmbeddingMLP import EmbeddingMLP

def train(epoch):
    for i, (xi, xv, y) in enumerate(loader_train):
        xi, xv, y = torch.squeeze(xi), torch.squeeze(xv), torch.squeeze(y).to(torch.float32)
        optimizer.zero_grad()
        out = model(xi, xv)
        loss = nn.BCELoss()(torch.squeeze(out, dim=1), y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch {}, i {}, loss {}".format(epoch, i, loss))



if __name__ == "__main__":
    train_path = 'data/raw/trainingSamples.csv'
    test_path = 'data/raw/testSamples.csv'
    continous_feature_names = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                               'userRatingCount', 'userAvgRating', 'userRatingStddev']
    categorial_feature_names = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                                'movieGenre1', 'movieGenre2', 'movieGenre3', 'userId', 'movieId']

    categorial_feature_vocabsize = [20] * 8 + [30001] + [1001]
    # build dataset for train and test
    train_data = build_dataset(train_path)
    loader_train = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = build_dataset(test_path)
    loader_test = DataLoader(test_data, batch_size=64, shuffle=True)
    # train model
    model = EmbeddingMLP(categorial_feature_vocabsize, continous_feature_names, categorial_feature_names)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    for i in range(100):
        train(i)



    # test_data = build_dataset(test_path)