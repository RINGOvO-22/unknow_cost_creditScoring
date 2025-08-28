"""Loading data from file"""
"""Modified based on performative_prediction_based/data_prep.py"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter


def load_data(file_loc, seed=None, test_size=0.2):
    """Load data from cvs file.

    Parameters
    ----------
        file_loc: string
            path to the '.cvs' training data file
    Returns
    -------
        X_full: np.array
            balances data matrix     
        Y_full: np.array
            corresponding labels (0/1) 
        data: DataFrame
            raw data     
    """

    data = pd.read_csv(file_loc, index_col=0)
    data.dropna(inplace=True)

    # full data set
    X_all = data.drop('SeriousDlqin2yrs', axis=1)

    # zero mean, unit variance
    X_all = preprocessing.scale(X_all)

    # add bias term
    X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)

    # outcomes
    Y_all = np.array(data['SeriousDlqin2yrs'])

    # balance classes
    default_indices = np.where(Y_all == 1)[0]
    other_indices = np.where(Y_all == 0)[0][:10000]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X_all[indices]
    Y_balanced = Y_all[indices]

    # shuffle arrays
    if seed is not None:
        np.random.seed(seed)
    p = np.random.permutation(len(indices))
    X_full = X_balanced[p]
    Y_full = Y_balanced[p]

    # split into training and test sets
    m = len(X_all)
    split = int((1-test_size)* m)

    return X_full[:split], Y_full[:split], X_full[split:], Y_full[split:]

if __name__ == "__main__":
    X, Y, data = load_data("data/GiveMeSomeCredit/cs-training.csv")
    print("X shape:", X.shape)
    print("Head of X:\n", X[:5])
    print("Y shape:", Y.shape)
    print(Counter(Y))
