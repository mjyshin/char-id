import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.decomposition import PCA


def dim_red(Xtrain, Xtest):
    # Run dimensionality reduction
    pca = PCA()
    pca.fit(Xtrain)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print("PCA results: min dimensions = ", d)
    pca = PCA(n_components=d)
    Xtrain_red = pca.fit_transform(Xtrain)
    Xtest_red = pca.transform(Xtest)
    return {'Xtrain_red': Xtrain_red, 'Xtest_red': Xtest_red}


def main():
    data = load_data()
    ytrain = data['ytrain']
    ytest = data['ytest']
    print(data['Xtest'].shape)  # Size of original test dataset

    X_red = dim_red(data['Xtrain'], data['Xtest'])
    Xtrain = X_red['Xtrain_red']
    Xtest = X_red['Xtest_red']
    print(Xtest.shape)  # Size of reduced test dataset


if __name__ == "__main__":
    main()
