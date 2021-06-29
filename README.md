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
The supported model are:
```
1. Embedding+MLP
2. WideDeep
3. NeuralCF
4. DeepFM
5. DIN
6. DIEN
```
if you want to retrain model DIN, you can use command line:
>  python3 DIN_example.py
# Implementated Network
* [DeepCross(Embedding+MLP)](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)
* [WideDeep](https://arxiv.org/pdf/1606.07792.pdf)
* [NeuralCF](https://arxiv.org/pdf/1708.05031v2.pdf)
* [DeepFM](https://arxiv.org/pdf/1703.04247.pdf)
* [DIN](https://arxiv.org/pdf/1706.06978.pdf)
* [DIEN](https://arxiv.org/pdf/1809.03672.pdf)


