import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import urllib.request, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def get_ticker_data(ticker):
    api_key = 'IQWSVHF8W4BLOHWY'
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = 'stock_market_data-%s.csv'%ticker
    filepath = os.path.dirname(os.path.realpath(__file__)) + '\\data\\' + file_to_save

    if not os.path.exists(filepath):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
            print('Data saved to: %s'%file_to_save)
            df.to_csv(filepath)

    data = pd.read_csv(filepath)
    data = data.sort_values('Date')

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    return data

class AverageLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(AverageLossCallback, self).__init__()
        self.total_loss = 0.0
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        self.total_loss += current_loss
        self.epoch_count += 1
        
        average_loss = self.total_loss / self.epoch_count

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_lstm(train_data,time_steps,batch_size,epochs):
    X_train, y_train = create_sequences(train_data, time_steps)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(time_steps, 1)))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=25))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    average_loss_callback = AverageLossCallback()

    model.fit(X_train, y_train, batch_size, epochs, callbacks=[average_loss_callback])

    return model

def predict_future(model, last_sequence, future_days):
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(future_days):
        current_sequence = current_sequence.reshape((1, time_steps, 1))
        future_prediction = model.predict(current_sequence)
        future_prediction = np.reshape(future_prediction, (1, 1, 1))
        current_sequence = np.append(current_sequence[:, 1:, :], future_prediction, axis=1)
        future_predictions.append(future_prediction[0, 0])

    return np.array(future_predictions).reshape(-1, 1)


ticker_list = ['PANW','V','ONTO','TSM','SNPS']
time_steps_list = [10, 20, 30, 40, 60]
epochs_list = [30,50,100]
batch_size_list = [16,32,64]

result = []
average_loss = -1
    
for ticker in ticker_list:
    for time_steps in time_steps_list:
        for epochs in epochs_list:
            for batch_size in batch_size_list:
                print(f"Training LSTM with ticker={ticker}, time_steps={time_steps}, batch_size={batch_size}, epochs={epochs}")
                data = get_ticker_data(ticker)

                close_prices = data['Close'].values
                close_prices = close_prices.reshape(-1,1)

                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(close_prices)

                train_size = int(len(scaled_data) * 0.8)
                test_size = len(scaled_data) - train_size
                train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

                X_test, y_test = create_sequences(test_data, time_steps)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                model = train_lstm(train_data,time_steps,batch_size,epochs)

                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)

                y_test = y_test.reshape(-1,1)
                y_test_inv = scaler.inverse_transform(y_test)

                mse_test = mean_squared_error(y_test_inv, predictions)

                last_sequence = test_data[-time_steps:]
                future_days = 30  
                future_predictions = predict_future(model, last_sequence, future_days)
                future_predictions_inv = scaler.inverse_transform(future_predictions)

                result.append({
                    'Ticker' : ticker,
                    'Time Steps' : time_steps,
                    'Epochs' : epochs,
                    'Batch Size' : batch_size,
                    'MSE' : mse_test,
                    'Price Prediction' : future_predictions_inv[-1],
                    'Average Loss' : average_loss
                })

df = pd.DataFrame(result)
df.to_csv('lstmdata.csv',index=False)
