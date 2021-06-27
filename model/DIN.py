import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import Sigmoid

class DIN(nn.Module):
    def __init__(self, candidate_movie_num, recent_rate_num, user_profile_num, context_feature_num, candidate_movie_dict, 
            recent_rate_dict, user_profile_dict, context_feature_dict, history_num, embed_dim, activation_dim, hidden_dim=[128, 64]):
        super().__init__()
        self.candidate_vocab_list = list(candidate_movie_dict.values())
        self.recent_rate_list = list(recent_rate_dict.values())
        self.user_profile_list = list(user_profile_dict.values())
        self.context_feature_list = list(context_feature_dict.values())
        self.embed_dim = embed_dim
        self.history_num = history_num
        # candidate_embedding_layer 
        self.candidate_embedding_list = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in self.candidate_vocab_list])
        # recent_rate_embedding_layer
        self.recent_rate_embedding_list = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in self.recent_rate_list])
        # user_profile_embedding_layer
        self.user_profile_embedding_list = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in self.user_profile_list])
        # context_embedding_list
        self.context_embedding_list = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for vocab_size in self.context_feature_list])

        # activation_unit
        self.activation_unit = nn.Sequential(nn.Linear(4*embed_dim, activation_dim), 
                                            nn.PReLU(),
                                            nn.Linear(activation_dim, 1),
                                            nn.Sigmoid())
        
        # self.dnn_part
        self.dnn_input_dim = len(self.candidate_embedding_list) * embed_dim + candidate_movie_num - len(
            self.candidate_embedding_list) + embed_dim + len(self.user_profile_embedding_list) * embed_dim + \
            user_profile_num - len(self.user_profile_embedding_list) + len(self.context_embedding_list) * embed_dim \
            + context_feature_num - len(self.context_embedding_list)

        self.dnn = nn.Sequential(nn.Linear(self.dnn_input_dim, hidden_dim[0]),
                             nn.BatchNorm1d(hidden_dim[0]),
                             nn.PReLU(),
                             nn.Linear(hidden_dim[0], hidden_dim[1]),
                             nn.BatchNorm1d(hidden_dim[1]),
                             nn.PReLU(),
                             nn.Linear(hidden_dim[1], 1),
                             nn.Sigmoid())

    def forward(self, candidate_features, recent_features, user_features, context_features):
        bs = candidate_features.shape[0]
        # candidate cate_feat embed
        candidate_embed_features = []
        for i, embed_layer in enumerate(self.candidate_embedding_list):
            candidate_embed_features.append(embed_layer(candidate_features[:, i].long()))
        candidate_embed_features = torch.stack(candidate_embed_features, dim=1).reshape(bs, -1).unsqueeze(1)
        ## add candidate continous feat
        candidate_continous_features = candidate_features[:, len(candidate_features):]
        candidate_branch_features = torch.cat([candidate_continous_features.unsqueeze(1), candidate_embed_features], dim=2).repeat(1, self.history_num, 1)

        # recent_rate  cate_feat embed
        recent_embed_features = []
        for i, embed_layer in enumerate(self.recent_rate_embedding_list):
            recent_embed_features.append(embed_layer(recent_features[:, i].long()))
        recent_branch_features = torch.stack(recent_embed_features, dim=1)
        
        # user_profile feat embed 
        user_profile_embed_features = []
        for i, embed_layer in enumerate(self.user_profile_embedding_list):
            user_profile_embed_features.append(embed_layer(user_features[:, i].long()))
        user_profile_embed_features = torch.cat(user_profile_embed_features, dim=1)
        ## add user_profile continous feat
        user_profile_continous_features = user_features[:, len(self.user_profile_list):]
        user_profile_branch_features = torch.cat([user_profile_embed_features, user_profile_continous_features], dim=1)

        # context embed feat
        context_embed_features = []
        for i, embed_layer in enumerate(self.context_embedding_list):
            context_embed_features.append(embed_layer(context_features[:, i].long()))
        context_embed_features = torch.cat(context_embed_features, dim=1)
        ## add context continous feat
        context_continous_features = context_features[:, len(self.context_embedding_list):]
        context_branch_features = torch.cat([context_embed_features, context_continous_features], dim=1)

        # activation_unit
        sub_unit_input = recent_branch_features - candidate_branch_features
        product_unit_input = torch.mul(recent_branch_features, candidate_branch_features)
        unit_input = torch.cat([recent_branch_features, candidate_branch_features, sub_unit_input, product_unit_input], dim=2)
        # weight-pool
        activation_unit_out = self.activation_unit(unit_input).repeat(1, 1, self.embed_dim)
        recent_branch_pooled_features = torch.mean(torch.mul(activation_unit_out, recent_branch_features), dim=1)
        # dnn part
        dnn_input = torch.cat([candidate_branch_features[:, 0, :], recent_branch_pooled_features, user_profile_branch_features, context_branch_features], dim=1)
        dnn_out = self.dnn(dnn_input)
        return dnn_out

