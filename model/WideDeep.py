import torch
import torch.nn as nn
import numpy as np

class WideDeep(nn.Module):
    def __init__(self, categorial_feature_vocabsize, continous_feature_names, categorial_feature_names,
                 device, embed_dim=10, hidden_dim=[128, 128]):
        super().__init__()
        assert len(categorial_feature_vocabsize) == len(categorial_feature_names)
        # deep part
        self.device = device
        # embedding layer
        self.embedding_layer_list = []
        for i in range(len(categorial_feature_vocabsize)):
            self.embedding_layer_list.append(nn.Embedding(categorial_feature_vocabsize[i], embed_dim))
        self.deep_mlp1 = nn.Linear(len(categorial_feature_vocabsize) * embed_dim + len(continous_feature_names),
                              hidden_dim[0])
        self.deep_bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.deep_relu1 = nn.ReLU(inplace=True)
        self.deep_mlp2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.deep_bn2 = nn.BatchNorm1d(hidden_dim[1])
        self.deep_relu2 = nn.ReLU(inplace=True)
        self.deep_mlp3 = nn.Linear(hidden_dim[1], 1)
        # wide part
        self.wide_mlp = nn.Linear(len(continous_feature_names), 1)
        self.act = nn.Sigmoid()

    def forward(self, xi, xv):
        # deep part
        embed_out_list = []
        for i, embed_layer in enumerate(self.embedding_layer_list):
            embed_out = embed_layer.to(self.device)(xv[:, i].long())
            embed_out_list.append(embed_out)
        xv = torch.cat(embed_out_list, dim=1)
        deep_x = torch.cat((xi, xv), dim=1)
        deep_x = self.deep_relu1(self.deep_bn1(self.deep_mlp1(deep_x)))
        deep_x = self.deep_relu2(self.deep_bn2(self.deep_mlp2(deep_x)))
        deep_x = self.deep_mlp3(deep_x)
        # wide part
        wide_x = self.wide_mlp(xi)
        x = self.act(0.5*wide_x+0.5*deep_x)
        return x