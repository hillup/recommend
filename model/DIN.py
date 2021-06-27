import torch
import torch.nn as nn
import numpy as np

class DIN(nn.Module):
    def __int__(self, candidate_movie_dict, recent_rate_dict, user_profile_dict, context_feature_dict, embed_dim):
        self.candidate_vocab_list = list(candidate_movie_dict.values())
        self.recent_rate_dict = list(recent_rate_dict.values())
        self.user_profile_dict = list(user_profile_dict.values())
        self.context_feature_dict = list(context_feature_dict.values())
        self.embed_dim = embed_dim
    def forward(self, candidate_columns):
        pass