import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.decomposition import PCA


def dim_red(Xtrain):
    pca = PCA()
    pca.fit(Xtrain)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print("PCA results: min dimensions = ", d, ", dimensions used = ", 100)
    pca = PCA(n_components=100)
    Xtrain_red = pca.fit_transform(Xtrain)
    Xtest_red = pca.transform(Xtest)
    X = {'Xtrain_red': Xtrain_red, 'Xtest_red': Xtest_red}


def main():
    data = load_data()
    print(data['Xtest'])


if __name__ == "__main__":
    main()
