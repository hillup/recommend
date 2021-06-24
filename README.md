# Recommend Model
recommend model using pytorch 
# Requirements
This is my experiment eviroument
* python3.6
* pytorch1.6.0+cu101
* pandas 1.2.4
# Usage
## 1. data
The data path in our experiment is :
* train_path: data/raw/trainingSamples.csv
* test_path: data/raw/testSamples.csv

data is from https://grouplens.org/datasets/movielens/, we sample som data from it.
## 2. train model
if you want to retrain model, you can use command line:
-   python3 train.py
