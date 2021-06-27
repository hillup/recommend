import torch
import torch.nn as nn
import numpy as np

class DeepFM(nn.Module):
    def __init__(self, categorial_feature_vocabsize, continous_feature_names, categorial_feature_names,
                 embed_dim=10, hidden_dim=[128, 128]):
        super().__init__()
        assert len(categorial_feature_vocabsize) == len(categorial_feature_names)
        self.continous_feature_names = continous_feature_names
        self.categorial_feature_names = categorial_feature_names
        # FM part
        # first-order part
        if continous_feature_names:
            self.fm_1st_order_dense = nn.Linear(len(continous_feature_names), 1)
        self.fm_1st_order_sparse_emb = nn.ModuleList([nn.Embedding(vocab_size, 1) for vocab_size in categorial_feature_vocabsize])
        # senond-order part
        self.fm_2nd_order_sparse_emb = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in categorial_feature_vocabsize])

        # deep part
        self.dense_embed = nn.Sequential(nn.Linear(len(continous_feature_names), len(categorial_feature_names)*embed_dim),
                                         nn.BatchNorm1d(len(categorial_feature_names)*embed_dim),
                                         nn.ReLU(inplace=True))
        self.dnn_part = nn.Sequential(nn.Linear(len(categorial_feature_vocabsize)*embed_dim, hidden_dim[0]),
                                        nn.BatchNorm1d(hidden_dim[0]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim[0], hidden_dim[1]),
                                        nn.BatchNorm1d(hidden_dim[1]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim[1], 1))
        # output act
        self.act = nn.Sigmoid()

    def forward(self, xi, xv):
        # FM first-order part
        fm_1st_sparse_res = []
        for i, embed_layer in enumerate(self.fm_1st_order_sparse_emb):
            fm_1st_sparse_res.append(embed_layer(xv[:, i].long()))
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)
        if xi is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(xi)
            fm_1st_part = fm_1st_dense_res + fm_1st_sparse_res
        else:
            fm_1st_part = fm_1st_sparse_res
        # FM second-order part
        fm_2nd_order_res = []
        for i, embed_layer in enumerate(self.fm_2nd_order_sparse_emb):
            fm_2nd_order_res.append(embed_layer(xv[:, i].long()))
        fm_2nd_concat_1d = torch.stack(fm_2nd_order_res, dim=1)   # [bs, n, emb_dim]
        # sum -> square
        square_sum_embed = torch.pow(torch.sum(fm_2nd_concat_1d, dim=1), 2)
        # square -> sum
        sum_square_embed = torch.sum(torch.pow(fm_2nd_concat_1d, 2), dim=1)
        # minus and half
        sub = 0.5 * (square_sum_embed - sum_square_embed)
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)
        # Dnn part
        dnn_input = torch.flatten(fm_2nd_concat_1d, 1)
        if xi is not None:
            dense_out = self.dense_embed(xi)
            dnn_input = dnn_input + dense_out
        dnn_output = self.dnn_part(dnn_input)
        out = self.act(fm_1st_part + fm_2nd_part + dnn_output)
        return out









