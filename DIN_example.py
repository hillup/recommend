import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import collections

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from data.dataset import build_din_dataset
from model.DIN import DIN

def train(epoch):
    model.train()
    for batch_idx, (x0, x1, x2, x3, y) in enumerate(loader_train):
        x0, x1, x2, x3, y = torch.squeeze(x0, 1).to(torch.float32), torch.squeeze(x1, 1).to(torch.float32), torch.squeeze(x2, 1).to(torch.float32), torch.squeeze(x3, 1).to(torch.float32), torch.squeeze(y).to(torch.float32)
        x0, x1, x2, x3, y = x0.to(device), x1.to(device), x2.to(device), x3.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x0, x1, x2, x3)
        loss = nn.BCELoss()(torch.squeeze(out, dim=1), y)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print("epoch {}, batch_idx {}, loss {}".format(epoch, batch_idx, loss))

def test(epoch, best_acc=0):
    model.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    for batch_idx, (x0, x1, x2, x3, y) in enumerate(loader_test):
        x0, x1, x2, x3, y = torch.squeeze(x0, 1).to(torch.float32), torch.squeeze(x1, 1).to(torch.float32), torch.squeeze(x2, 1).to(torch.float32), torch.squeeze(x3, 1).to(torch.float32), torch.squeeze(y).to(torch.float32)
        x0, x1, x2, x3, y = x0.to(device), x1.to(device), x2.to(device), x3.to(device), y.to(device)
        out = model(x0, x1, x2, x3)
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
    parser.add_argument('-save_path', action='store_true', default='checkpoint/DIN/DIN_best.pth',
                        help='save model path')

    args = parser.parse_args()
    candidate_movie_dict = collections.OrderedDict(movieId=1001)
    recent_rate_dict = collections.OrderedDict(userRatedMovie1=1001, userRatedMovie2=1001, userRatedMovie3=1001,
                                               userRatedMovie4=1001, userRatedMovie5=1001)
    user_profile_dict = collections.OrderedDict(userId=30001, userGenre1=20, userGenre2=20, userGenre3=20, userGenre4=20,
                                                userGenre5=20)
    context_feature_dict = collections.OrderedDict(movieGenre1=20, movieGenre2=20, movieGenre3=20)
    candidate_movie_col = ['movieId']
    recent_rate_col = ['userRatedMovie1', 'userRatedMovie2', 'userRatedMovie3', 'userRatedMovie4', 'userRatedMovie5']
    user_profile_col = ['userId', 'userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5', 'userRatingCount',
                    'userAvgRating', 'userRatingStddev']
    context_features_col = ['movieGenre1', 'movieGenre2', 'movieGenre3', 'releaseYear', 'movieRatingCount',
                        'movieAvgRating', 'movieRatingStddev']
    # categorial_feature_vocabsize = [20] * 8 + [30001] + [1001]
    # build dataset for train and test
    batch_size = args.bs
    train_data = build_din_dataset(args.train_path)
    loader_train = DataLoader(train_data, batch_size=batch_size, num_workers=256, shuffle=True, pin_memory=True)
    test_data = build_din_dataset(args.test_path)
    loader_test = DataLoader(test_data, batch_size=batch_size, num_workers=256)

    device = torch.device("cuda" if args.gpu else "cpu")
    # train model
    model = DIN(len(candidate_movie_col), len(recent_rate_col), len(user_profile_col), len(context_features_col), 
        candidate_movie_dict, recent_rate_dict, user_profile_dict, context_feature_dict, 5, 20, 32)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_acc = 0
    for ep in range(args.epoches):
        train(ep)
        best_acc = test(ep, best_acc)
