# test_misc.py
import re
import io
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

from misc import load_data, split_data, train_and_evaluate

def test_load_data_columns_and_types():
    df = load_data()
    # Feature names from spec
    expected_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    assert isinstance(df, pd.DataFrame), "load_data should return a pandas DataFrame"
    assert list(df.columns) == expected_cols, f"Unexpected columns. Got {list(df.columns)}"
    # Basic sanity checks
    assert df.shape[0] > 0, "DataFrame should have at least one row"
    assert df.isnull().sum().sum() == 0, "DataFrame should not contain NaNs"

def test_split_data_shapes_and_types():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    # types
    assert hasattr(X_train, 'shape') and hasattr(X_test, 'shape')
    assert len(X_train.columns) == 13, "Expected 13 features"
    # sizes
    total = X_train.shape[0] + X_test.shape[0]
    assert total == df.shape[0], "Train+Test split should equal total rows"
    assert y_train.ndim == 1 or len(y_train.shape) == 1

def test_train_and_evaluate_decision_tree_returns_non_negative_mse():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = DecisionTreeRegressor(random_state=42)
    mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    assert isinstance(mse, float), "MSE should be a float"
    assert mse >= 0.0, "MSE should be non-negative"

def test_train_and_evaluate_kernel_ridge_returns_non_negative_mse():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = KernelRidge()
    mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    assert isinstance(mse, float)
    assert mse >= 0.0
