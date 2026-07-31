# src/preprocess.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import RANDOM_STATE, TARGET_COL


def load_data(csv_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(csv_path)
    return df


def split_data(df):
    """
    Split dataset into train and test sets.
    """

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):
    """
    Create preprocessing pipeline.
    """

    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    # Numerical Pipeline
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical Pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Column Transformer
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocess