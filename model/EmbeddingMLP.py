import torch
import numpy as np
import torch.nn as nn

class EmbeddingMLP(nn.Module):
    def __init__(self, categorial_feature_vocabsize, continous_feature_names, categorial_feature_names, embed_dim=10, hidden_dim=[128, 128]):
        super().__init__()
        assert len(categorial_feature_vocabsize) == len(categorial_feature_names)
        # embedding layer
        self.embedding_layer_list = []
        for i in range(len(categorial_feature_vocabsize)):
            embed_layer = nn.Embedding(categorial_feature_vocabsize[i], embed_dim)
            self.embedding_layer_list.append(embed_layer)
        self.mlp1 = nn.Linear(len(categorial_feature_vocabsize)*embed_dim+len(continous_feature_names), hidden_dim[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.mlp2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.mlp3 = nn.Linear(hidden_dim[1], 1)
        self.act = nn.Sigmoid()

    def forward(self, xi, xv):
        embed_out_list = []
        for i, embed_layer in enumerate(self.embedding_layer_list):
            embed_out = embed_layer(xv[:, i].long())
            embed_out_list.append(embed_out)
        xv = torch.cat(embed_out_list, dim=1)
        x = torch.cat((xi, xv), dim=1).to(torch.float32)
        x = self.relu1(self.mlp1(x))
        x = self.relu2(self.mlp2(x))
        x = self.act(self.mlp3(x))
        return x
