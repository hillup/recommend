import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from data.dataset import build_dataset
from model.DeepFM import DeepFM

def train(epoch):
    model.train()
    for batch_idx, (xi, xv, y) in enumerate(loader_train):
        xi, xv, y = torch.squeeze(xi).to(torch.float32), torch.squeeze(xv), torch.squeeze(y).to(torch.float32)
        if args.gpu:
            xi, xv, y = xi.to(device), xv.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(xi, xv)
        loss = nn.BCELoss()(torch.squeeze(out, dim=1), y)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print("epoch {}, batch_idx {}, loss {}".format(epoch, batch_idx, loss))

def test(epoch, best_acc=0):
    model.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    for batch_idx, (xi, xv, y) in enumerate(loader_test):
        xi, xv, y = torch.squeeze(xi).to(torch.float32), torch.squeeze(xv), torch.squeeze(y).to(torch.float32)
        if args.gpu:
            xi, xv, y = xi.to(device), xv.to(device), y.to(device)
        out = model(xi, xv)
        test_loss += nn.BCELoss()(torch.squeeze(out, dim=1), y).item()
        correct += ((torch.squeeze(out, dim=1) > 0.5) == y).sum().item()
    if correct/len(loader_test) > best_acc:
        best_acc = correct/len(loader_test)
        torch.save(model, args.save_path)
    print("epoch {}, test loss {}, test acc {}".format(epoch, test_loss/len(loader_test), correct/len(loader_test)))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-bs', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-epoches', type=int, default=15, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-train_path', action='store_true', default='data/raw/trainingSamples.csv',
                        help='train data path')
    parser.add_argument('-test_path', action='store_true', default='data/raw/testSamples.csv',
                        help='test data path')
    parser.add_argument('-save_path', action='store_true', default='checkpoint/DeepFM/DeepFm_best.pth',
                        help='save model path')

    args = parser.parse_args()

    continous_feature_names = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                               'userRatingCount', 'userAvgRating', 'userRatingStddev']
    categorial_feature_names = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                                'movieGenre1', 'movieGenre2', 'movieGenre3', 'userId', 'movieId']

    categorial_feature_vocabsize = [20] * 8 + [30001] + [1001]
    # build dataset for train and test
    batch_size = args.bs
    train_data = build_dataset(args.train_path)
    loader_train = DataLoader(train_data, batch_size=batch_size, num_workers=64, shuffle=True, pin_memory=True)
    test_data = build_dataset(args.test_path)
    loader_test = DataLoader(test_data, batch_size=batch_size, num_workers=64)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if args.gpu else "cpu")
    # train model
    model = DeepFM(categorial_feature_vocabsize, continous_feature_names, categorial_feature_names, embed_dim=64)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_acc = 0
    for ep in range(args.epoches):
        train(ep)
        best_acc = test(ep, best_acc)
