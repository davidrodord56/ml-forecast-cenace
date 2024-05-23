import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_true: numpy array or list, true target values.
    - y_pred: numpy array or list, predicted target values.

    Returns:
    - float, RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
    - y_true: numpy array or list, true target values.
    - y_pred: numpy array or list, predicted target values.

    Returns:
    - float, MAPE.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return sum(np.abs((y_true - y_pred) / y_true)) * (1/ len(y_true)) * 100

def calculate_correlation(y_true, y_pred):
    """
    Calculate the correlation between true and predicted values using Pearson's correlation coefficient.

    Parameters:
    - y_true: numpy array or list, true target values.
    - y_pred: numpy array or list, predicted target values.

    Returns:
    - float, correlation coefficient.
    """
    return pearsonr(y_true, y_pred)[0]


