import torch
import numpy as np
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embed_dim=10, hidden_dim=[128, 128]):
        super().__init__()
        self.user_embeddding_layer = nn.Embedding(user_feature_dim, embed_dim)
        self.item_embedding_layer = nn.Embedding(item_feature_dim, embed_dim)

        # inter layers
        self.inter_layers = []
        input_dim = 2*embed_dim
        for dim in hidden_dim:
            self.inter_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim
        self.inter_layers = nn.ModuleList(self.inter_layers)
        self.linear = nn.Linear(hidden_dim[-1], 1)
        self.act = nn.Sigmoid()

    def forward(self, user_feature, item_feature):
        user_feature = self.user_embeddding_layer(user_feature.long())
        item_feature = self.item_embedding_layer(item_feature.long())

        x = torch.cat((user_feature, item_feature), dim=1)
        for layer in self.inter_layers:
            x = layer(x)
        x = self.act(self.linear(x))
        return x
