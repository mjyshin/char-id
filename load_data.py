from PIL import Image
import pandas as pd
import numpy as np


def load_data():
    """
    LOAD DATASETS
    Inputs:     20 x 20 color .bmp images
    Processes:  Convert each 20 x 20 RGB image to 1 x 400 grayscale array
                Split data and labels between train and test sets
    Outputs:    Datasets including train data and labels, and test data and labels
    """

    # Dataset dimensions
    N = 6283

    # Load feature dataset and convert to grayscale
    X = []
    for i in range(1, N + 1):
        img = Image.open("./trainResized/{}.Bmp".format(str(i)))
        img = img.convert("LA")
        imgdata = list(img.getdata())
        img = []
        for value in imgdata:
            img.append(value[0] / value[1])
        X.append(img)
    X = pd.DataFrame(X)

    # Load label dataset
    yreal = pd.read_csv("trainLabels.csv", usecols=[1])
    yclass = yreal.iloc[0:N]
    y = yclass.copy()
    for i in range(0, N):
        y.iloc[i] = ord(str(yclass.iloc[i].values)[2])

    # Divide training/test sets
    Ntrain = 5000

    def split_dataset(data, Ntrain):
        np.random.seed(0)
        Ntest = int(len(data) - Ntrain)
        shuffled_ind = np.random.permutation(len(data))
        train_ind = shuffled_ind[Ntest:]
        test_ind = shuffled_ind[:Ntest]
        return data.iloc[train_ind].reset_index(drop=True), data.iloc[test_ind].reset_index(drop=True)

    Xtrain, Xtest = split_dataset(X, Ntrain)
    ytrain, ytest = split_dataset(y, Ntrain)
    ytrain = ytrain.values.ravel()
    ytest = ytest.values.ravel()

    return {'Xtrain': Xtrain, 'ytrain': ytrain, 'Xtest': Xtest, 'ytest': ytest}
