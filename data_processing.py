import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    recall_score,
    f1_score,
    auc
)

import warnings
warnings.filterwarnings("ignore")

from sklearn import set_config

def get_data_matrices():
    set_config(transform_output="pandas")

    df = pd.read_csv("data.csv")
    df.info()

    df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    df["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    corr_matrix = df.drop("diagnosis", axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(to_drop, axis=1, inplace=True)
    print("Dropped correlated columns:", to_drop)

    corr = df.corr()
    sorted_corr = corr.reindex(corr["diagnosis"].abs().sort_values(ascending=False).index, axis=1)

    corr = df.corr()
    sorted_corr = corr.reindex(corr["diagnosis"].abs().sort_values(ascending=False).index, axis=1)

    shuffled_df = df.sample(frac=1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        shuffled_df.drop("diagnosis", axis=1),
        shuffled_df[["diagnosis"]],
        test_size=0.2, shuffle=False
    )

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y=y_train)
    X_test = scaler.transform(X_test)

    largest_singular_value = np.linalg.norm(X_train, ord=2)
    X_train = X_train / largest_singular_value

    print("Train shape:", X_train.shape)
    X_train.info()

    return X_train, X_test, y_train, y_test

def get_Lipschitz_constant(X_train):
    # Lipschitz constant, spectral norm
    L = np.linalg.norm(X_train.T @ X_train, ord=2)
    print("Lipschitz constant:", L)
    return L