from data_retrieval.collector import collect_data_api
from data_retrieval.collector import aggregate_data
from model.ml_model import data_to_seq2seq
import time
import joblib
import polars as pl
import tensorflow as tf

def data_prepare_last_week():
    start_epoch = int(time.time()) - 8*24*60*60
    # print(start_epoch)
    # data = collect_data_api(system="NAC", start_epoch= start_epoch, end_epoch= 9664385602)
    # print(data)
    # data = aggregate_data(data)
    # print(data)
    # scaler = joblib.load("./model/tran_scaler.save")
    # print(scaler)
    # data_to_model = data.select(pl.col(['demand']))
    # print(data_to_model)
    # data_to_model =  pl.DataFrame(scaler.transform(data_to_model))
    # print(data_to_model)
    #
    #
    # X = data_to_model[:-24]
    # print(X)
    #
    #
    # model = tf.keras.models.load_model("./model/Model-12-03-2024-13-30-MAE0.03.keras")
    # model.summary()
    #
    #
    # y_pred = model.predict(X)[0,-1]
    # print(y_pred)
    # print("**********")
    # print(data_to_model[-24:])





print(data_prepare_last_week())