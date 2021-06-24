import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from data.dataset import build_dataset
from model.EmbeddingMLP import EmbeddingMLP

def train(epoch):
    model.train()
    for batch_idx, (xi, xv, y) in enumerate(loader_train):
        xi, xv, y = torch.squeeze(xi).to(torch.float32), torch.squeeze(xv), torch.squeeze(y).to(torch.float32)
        if args.gpu:
            xi, xv, y = xi.cuda(), xv.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(xi, xv)
        loss = nn.BCELoss()(torch.squeeze(out, dim=1), y)
        loss.backward()
        optimizer.step()
        print("epoch {}, batch_idx {}, loss {}".format(epoch, batch_idx, loss))

def test(epoch):
    model.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    for batch_idx, (xi, xv, y) in enumerate(loader_test):
        xi, xv, y = torch.squeeze(xi).to(torch.float32), torch.squeeze(xv), torch.squeeze(y).to(torch.float32)
        if args.gpu:
            xi, xv, y = xi.cuda(), xv.cuda(), y.cuda()
        out = model(xi, xv)
        test_loss += nn.BCELoss()(torch.squeeze(out, dim=1), y).item()
        correct += ((torch.squeeze(out, dim=1) > 0.5) == y).sum().item()
    print("epoch {}, test loss {}, test acc {}".format(epoch, test_loss/len(loader_test), correct/len(loader_test)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-bs', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-epoches', type=int, default=15, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-train_path', action='store_true', default='data/raw/trainingSamples.csv',
                        help='train data path')
    parser.add_argument('-test_path', action='store_true', default='data/raw/testSamples.csv',
                        help='test data path')

    args = parser.parse_args()

    continous_feature_names = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                               'userRatingCount', 'userAvgRating', 'userRatingStddev']
    categorial_feature_names = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                                'movieGenre1', 'movieGenre2', 'movieGenre3', 'userId', 'movieId']

    categorial_feature_vocabsize = [20] * 8 + [30001] + [1001]
    # build dataset for train and test
    batch_size = args.bs
    train_data = build_dataset(args.train_path)
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = build_dataset(args.test_path)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train model
    model = EmbeddingMLP(categorial_feature_vocabsize, continous_feature_names, categorial_feature_names)
    if args.gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    for ep in range(args.epoches):
        train(ep)
        test(ep)
