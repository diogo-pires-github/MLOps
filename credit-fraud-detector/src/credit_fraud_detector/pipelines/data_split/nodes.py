import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split

def split_data(df: pd.DataFrame):
    print('No Frauds', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')

    X = df.drop('Class', axis=1)
    y = df['Class']

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    _, train_counts_label = np.unique(original_ytrain, return_counts=True)
    _, test_counts_label = np.unique(original_ytest, return_counts=True)
    print('-' * 100)

    print('Label Distributions: \n')
    print(train_counts_label / len(original_ytrain))
    print(test_counts_label / len(original_ytest))

    return (original_Xtrain, original_Xtest, original_ytrain, original_ytest)

def split_data2(
    df: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    y = df[parameters["target_column"]]
    X = df.drop(columns=parameters["target_column"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=parameters["test_fraction"], random_state=parameters["random_state"])
    return X_train, X_test, y_train, y_test
