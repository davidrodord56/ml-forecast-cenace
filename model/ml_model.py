from data_retrieval.collector import collect_data_api, aggregate_data
from sklearn.preprocessing import MinMaxScaler
import polars as pl
import tensorflow as tf
import joblib
import numpy as np
from datetime import datetime

def data_processing(data):
    targets = data.select(pl.col(['demand']))
    scaler = MinMaxScaler()
    data = pl.DataFrame(scaler.fit_transform(targets))
    scaler_filename =  "tran_scaler.save"
    joblib.dump(scaler, scaler_filename)
    return data
def data_split(data, ratio=80):
    lenght = len(data)
    train = int((ratio/100)*lenght)
    validation = int((lenght - train)*.8)
    test = lenght - train - validation

    data_train = data[0:train]
    data_valid = data[train:train + validation]
    data_test = data[train + validation:]


    return np.array(data_train), np.array(data_valid), data_test
def data_to_seq2seq(data_train,data_valid):
    seq2seq_train = to_seq2seq_dataset(data_train, shuffle=True, seed=42)
    seq2seq_valid = to_seq2seq_dataset(data_valid)
    return seq2seq_train,seq2seq_valid
def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return  dataset.flat_map(lambda window_ds: window_ds.batch(length))
def to_seq2seq_dataset(series, seq_length=168, ahead=24, target_col=1,
                       batch_size=32, shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:]))


    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)


    return ds.batch(batch_size)
def gen_model():
    tf.random.set_seed(42)  # extra code – ensures reproducibility
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(168, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.Dense(168, activation='tanh'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(48, return_sequences=True),
        tf.keras.layers.GRU(48, return_sequences=True),
        tf.keras.layers.Dense(24, activation='elu')
    ])
    model.summary()
    return model


def get_filename(valid_mae):
    now = datetime.now()
    now = now.strftime("%d-%m-%Y-%H-%M")
    filename = f"Model-{now}-MAE{round(valid_mae, 2)}.keras"
    return filename

def fit_train(model, train_set, valid_set, learning_rate, epochs=300, store = False):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=30, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum = 0.9)
    model.compile(loss=tf.losses.Huber(), optimizer=opt,metrics=["mae"])
    model.fit(train_set, validation_data=valid_set, epochs=epochs,callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    print(f"Model Trained : ValidationMAE{valid_mae}")



    if store:
        model.save(get_filename(valid_mae))
    return model



import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
print(len(devices))


# all epoch = 9794444444 / Normal = 1794444444
newdata = collect_data_api(end_epoch=9794444444)
aggdata = aggregate_data(newdata)
procdata = data_processing(aggdata)
data_train, data_valid, data_test = data_split(procdata)
seq2seq_train, seq2seq_valid = data_to_seq2seq(data_train,data_valid)
seq2seq_model = gen_model()
fit_train(seq2seq_model, seq2seq_train, seq2seq_valid,learning_rate=0.3, store=True)







