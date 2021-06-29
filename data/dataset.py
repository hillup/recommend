from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import json
import os

gnere_dict = {'Film-Noir': 0, 'Action': 1, 'Adventure': 2, 'Horror': 3, 'Romance': 4, 'War': 5, 'Comedy': 6, 'Western': 7,
              'Documentary': 8, 'Sci-Fi': 9, 'Drama': 10, 'Thriller': 11, 'Crime': 12, 'Fantasy': 13, 'Animation': 14,
              'IMAX': 15, 'Mystery': 16, 'Children': 17, 'Musical': 18, 0: 19}

class build_dataset(Dataset):
    def __init__(self, root, continous_feature_names=['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
        'userRatingCount', 'userAvgRating', 'userRatingStddev'], categorial_feature_names=['userGenre1', 'userGenre2',
        'userGenre3', 'userGenre4', 'userGenre5', 'movieGenre1', 'movieGenre2', 'movieGenre3', 'userId', 'movieId']):
        self.root = root
        self.data = pd.read_csv(root)
        self.data = self.data.fillna(0)
        self.continous_feature_names = continous_feature_names
        self.categorial_feature_names = categorial_feature_names
        

    def __getitem__(self, idx):
        idx = [idx]
        continous_features = self.data.iloc[idx][self.continous_feature_names]
        categorical_features = self.data.iloc[idx][self.categorial_feature_names[:-2]]
        for feature_name in self.categorial_feature_names[:-2]:
            categorical_features[feature_name] = categorical_features[feature_name].map(gnere_dict)
        continous_features = continous_features.values
        categorical_features = categorical_features.values
        userId_features = self.data.iloc[idx]['userId'].values.reshape(-1, 1)
        movieId_features = self.data.iloc[idx]['movieId'].values.reshape(-1, 1)
        categorical_features = np.concatenate((categorical_features, userId_features, movieId_features), axis=1).astype(np.int32)
        label = self.data.iloc[idx]['label'].values.reshape(-1, 1)
        return torch.from_numpy(continous_features), torch.from_numpy(categorical_features), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

class build_din_dataset(Dataset):
    def __init__(self, root, candidate_movie_col=['movieId'], recent_rate_col=['userRatedMovie1', 'userRatedMovie2',
        'userRatedMovie3', 'userRatedMovie4', 'userRatedMovie5'], user_profile=['userId', 'userGenre1', 'userGenre2',
        'userGenre3', 'userGenre4', 'userGenre5', 'userRatingCount', 'userAvgRating', 'userRatingStddev'], context_features=
    ['movieGenre1', 'movieGenre2', 'movieGenre3', 'releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev']):
        self.root = root
        self.data = pd.read_csv(self.root)
        self.data = self.data.fillna(0)
        self.candidate_movie_col = candidate_movie_col
        self.recent_rate_col = recent_rate_col
        self.user_profile_col = user_profile
        self.context_feature_col = context_features
        for feature_name in self.user_profile_col[1:6]:
            self.data[feature_name] = self.data[feature_name].fillna(0).map(gnere_dict)
        for feature_name in self.context_feature_col[:3]:
            self.data[feature_name] = self.data[feature_name].map(gnere_dict)

    def __getitem__(self, idx):
        idx = [idx]
        candidate_movie_features = self.data.iloc[idx][self.candidate_movie_col].values
        recent_rate_features = self.data.iloc[idx][self.recent_rate_col].astype(int).values
        user_profile_features = self.data.iloc[idx][self.user_profile_col].values
        context_features = self.data.iloc[idx][self.context_feature_col].values
        label = self.data.iloc[idx]['label'].values.reshape(-1, 1)
        return torch.from_numpy(candidate_movie_features), torch.from_numpy(recent_rate_features), \
               torch.from_numpy(user_profile_features), torch.from_numpy(context_features), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dd = build_din_dataset('raw/trainingSamples.csv')
    
