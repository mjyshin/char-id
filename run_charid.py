import numpy as np
import time
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


def dim_red(Xtrain, Xtest):
    """
    RUN DIMENSIONALITY REDUCTION

    Inputs:     Full train and test data (n x 400)
    Processes:  Reduce dimension from 400 to d while maintaining 95% of data variance
    Outputs:    Reduced train and test data (n x d)
    """
    pca = PCA()
    pca.fit(Xtrain)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print("PCA results: min dimensions = ", d)
    pca = PCA(n_components=d)
    Xtrain_red = pca.fit_transform(Xtrain)
    Xtest_red = pca.transform(Xtest)
    return {'Xtrain_red': Xtrain_red, 'Xtest_red': Xtest_red}


def run_svm(Xtrain, ytrain, Xtest, ytest):
    """
    RUN SUPPORT VECTOR MACHINE

    Inputs:     Reduced train and test data and labels
    Processes:  Run cross validation with SVM on train data and run best model on test data
    Outputs:    Optimal SVM model
    """

    print("\n***** SUPPORT VECTOR MACHINE *****")
    rbf_svm_best_mean = 0.
    rbf_svm_best_param = [0., 0.]
    cv_mean_acc = np.zeros((3, 10))
    i = -1
    for g in np.logspace(-3, -1, 3):
        i = i + 1
        j = -1
        for c in np.linspace(3.5, 5, 10):
            j = j + 1
            rbf_svm_clf = Pipeline((
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", gamma=g, C=c))
            ))
            time_start = time.clock()
            cv_accuracies = cross_val_score(rbf_svm_clf, Xtrain, ytrain, cv=5, scoring="accuracy")
            time_elapsed = time.clock() - time_start
            cv_mean_acc[i, j] = np.mean(cv_accuracies)
            print("CV mean = ", cv_mean_acc[i, j], ", CV training time = ", time_elapsed)
            if cv_mean_acc[i, j] > rbf_svm_best_mean:
                rbf_svm_best_mean = cv_mean_acc[i, j]
                rbf_svm_best_param = [g, c]
    # sio.savemat('rbf_svm_cv_mean_acc.mat', {'cv_mean_acc': cv_mean_acc})
    print("CV results: gamma = ", rbf_svm_best_param[0], ", C = ", rbf_svm_best_param[1])
    rbf_svm_best = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=rbf_svm_best_param[0], C=rbf_svm_best_param[1], probability=True))
    ))
    time_start = time.clock()
    rbf_svm_best.fit(Xtrain, ytrain)
    time_elapsed = time.clock() - time_start
    ytrain_pred = rbf_svm_best.predict(Xtrain)
    train_accuracy = accuracy_score(ytrain, ytrain_pred)
    print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
    ytest_pred = rbf_svm_best.predict(Xtest)
    test_accuracy = accuracy_score(ytest, ytest_pred)
    print("Test accuracy = ", test_accuracy)

    return rbf_svm_best


def run_knn(Xtrain, ytrain, Xtest, ytest):
    """
    RUN K-NEAREST NEIGHBORS

    Inputs:     Reduced train and test data and labels
    Processes:  Run cross validation with kNN on train data and run best model on test data
    Outputs:    Optimal kNN model
    """

    print("\n***** K-NEAREST NEIGHBORS *****")
    knn_best_mean = 0.
    knn_best_param = ['uniform', 1]
    cv_mean_acc = np.zeros((2, 10))
    i = -1
    for weigh in ['uniform', 'distance']:
        i = i + 1
        j = -1
        for neigh in range(1, 11):
            j = j + 1
            knn_clf = KNeighborsClassifier(weights=weigh, n_neighbors=neigh)
            time_start = time.clock()
            cv_accuracies = cross_val_score(knn_clf, Xtrain, ytrain, cv=5, scoring="accuracy")
            time_elapsed = time.clock() - time_start
            cv_mean_acc[i, j] = np.mean(cv_accuracies)
            print("CV mean accuracy = ", cv_mean_acc[i, j], ", CV training time = ", time_elapsed)
            if cv_mean_acc[i, j] > knn_best_mean:
                knn_best_mean = cv_mean_acc[i, j]
                knn_best_param = [weigh, neigh]
    # sio.savemat('knn_cv_mean_acc.mat', {'cv_mean_acc': cv_mean_acc})
    print("CV results: weights = ", knn_best_param[0], ", n_neighbors = ", knn_best_param[1])
    knn_best = KNeighborsClassifier(weights=knn_best_param[0], n_neighbors=knn_best_param[1])
    time_start = time.clock()
    knn_best.fit(Xtrain, ytrain)
    time_elapsed = time.clock() - time_start
    ytrain_pred = knn_best.predict(Xtrain)
    train_accuracy = accuracy_score(ytrain, ytrain_pred)
    print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
    ytest_pred = knn_best.predict(Xtest)
    test_accuracy = accuracy_score(ytest, ytest_pred)
    print("Test accuracy = ", test_accuracy)

    return knn_best


def run_voting(Xtrain, ytrain, Xtest, ytest, rbf_svm_best, knn_best):
    """
    RUN VOTING CLASSIFIER

    Inputs:     Reduced train and test data and labels and optimal SVM and kNN models
    Processes:  Run voting classification on test data
    Outputs:    None
    """

    print("\n***** SOFT VOTING *****")
    voting_clf = VotingClassifier(
        estimators=[('svc', rbf_svm_best), ('knn', knn_best)],
        voting='soft'
    )
    time_start = time.clock()
    voting_clf.fit(Xtrain, ytrain)
    time_elapsed = time.clock() - time_start
    ytrain_pred = voting_clf.predict(Xtrain)
    train_accuracy = accuracy_score(ytrain, ytrain_pred)
    print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
    ytest_pred = voting_clf.predict(Xtest)
    test_accuracy = accuracy_score(ytest, ytest_pred)
    print("Test accuracy = ", test_accuracy)


def main():
    data = load_data()
    ytrain = data['ytrain']
    ytest = data['ytest']
    print(data['Xtest'].shape)  # Size of original test dataset

    X_red = dim_red(data['Xtrain'], data['Xtest'])
    Xtrain = X_red['Xtrain_red']
    Xtest = X_red['Xtest_red']
    print(Xtest.shape)  # Size of reduced test dataset

    rbf_svm_best = run_pca(Xtrain, ytrain, Xtest, ytest)
    knn_best = run_knn(Xtrain, ytrain, Xtest, ytest)
    run_voting(Xtrain, ytrain, Xtest, ytest, rbf_svm_best, knn_best)


if __name__ == "__main__":
    main()
