import pandas as pd

from data_retrieval.collector import collect_data_api
from data_retrieval.collector import aggregate_data
from model.ml_model import data_to_seq2seq
import time
import joblib
import polars as pl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hvplot
from metrics import calculate_correlation, calculate_mape,calculate_rmse




def data_prepare_last_week():
    print("This is predictor")
    start_epoch = int(time.time()) - 48*24*60*60 + 1 #Regresar a 24
    end_epoch = start_epoch + 8*24*60*60

    data = collect_data_api(system="NAC", start_epoch= start_epoch, end_epoch= end_epoch)

    data = aggregate_data(data)
    print("Original Data")
    print(data)
    scaler = joblib.load("../model/tran_scaler.save")

    data_to_model = data.select(pl.col(['demand']))

    data_to_model =  pl.DataFrame(scaler.transform(data_to_model))



    X = data_to_model[:-24]
    print("Data to input model")
    print(X)

    model = tf.keras.models.load_model("../model/Model-12-03-2024-19-38-MAE0.04.keras")
    # model.summary()

    y_pred = model.predict(X.to_numpy()[np.newaxis, :-24])[0,-1]
    print(y_pred.shape)
    print("**********")
    print(data_to_model[-24:].to_numpy()[:,0].shape)
    # a = y_pred - data_to_model[-24:].to_numpy()
    print("**********")
    print(sum(abs(y_pred-data_to_model[-24:].to_numpy()[:,0])))

    print(y_pred)
    print(data_to_model[-24:].to_numpy()[:,0])

    # plt.plot(np.arange(0, 24), data_to_model[-24:].to_numpy()[:,0], label ="Real")
    # plt.plot(np.arange(0, 24), y_pred, label = "Prediction")
    # plt.legend(loc="upper left")
    # plt.show()

    real = pl.DataFrame(data.select(pl.col(['datetime','demand']))[-24:])
    prediction = pl.DataFrame(scaler.inverse_transform(y_pred.reshape(-1,1)), schema=['Prediction'])
    final =  pl.concat([real,prediction], how='horizontal')
    print(final)
    final_pd = final.to_pandas()
    final_pd['datetime'] = pd.to_datetime(final_pd['datetime'])
    final_pd = final_pd.set_index('datetime')
    print(final_pd)
    final_pd.plot(ylim=(0,52000))

    plt.show()

    true_values = final_pd['demand']
    predicted_values = final_pd['Prediction']

    rmse = calculate_rmse(true_values, predicted_values)
    mape = calculate_mape(true_values, predicted_values)
    correlation = calculate_correlation(true_values, predicted_values)

    print(f"RMSE: {round(rmse, 7)}")
    print(f"MAPE: {round(mape, 7)}%")
    print(f"Correlation: {round(correlation, 7)}")



data_prepare_last_week()