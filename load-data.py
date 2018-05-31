from PIL import Image
import pandas as pd
import numpy as np

def load_data()
    # Dataset dimensions
    N = 6283
    D = 400

    # Load feature dataset and convert to grayscale
    X = []
    for i in range(1, N + 1):
        img = Image.open("./trainResized/{}.Bmp".format(str(i)))
        img = img.convert("LA")
        imgData = list(img.getdata())
        img = []
        for value in imgData:
            img.append(value[0] / value[1])
        X.append(img)
    X = pd.DataFrame(X)

    # Load label dataset
    yReal = pd.read_csv("trainLabels.csv", usecols=[1])
    yClass = yReal.iloc[0:N]
    y = yClass.copy()
    for i in range(0, N):
        y.iloc[i] = ord(str(yClass.iloc[i].values)[2])

    # Divide training/test sets
    Ntrain = 5000

    def split_dataset(data, trainSize):
        np.random.seed(0)
        testSize = int(len(data) - trainSize)
        shuffled_ind = np.random.permutation(len(data))
        train_ind = shuffled_ind[testSize:]
        test_ind = shuffled_ind[:testSize]
        return data.iloc[train_ind].reset_index(drop=True), data.iloc[test_ind].reset_index(drop=True)

    Xtrain, Xtest = split_dataset(X, Ntrain)
    ytrain, ytest = split_dataset(y, Ntrain)
    ytrain = ytrain.values.ravel()
    ytest = ytest.values.ravel()

    return {'Xtrain':Xtrain, 'ytrain':ytrain, 'Xtest':Xtrain, 'ytest':ytest}