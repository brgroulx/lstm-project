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

api_key = 'IQWSVHF8W4BLOHWY'

ticker = 'HPQ'

url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key) # (json)

file_to_save = 'stock_market_data-%s.csv'%ticker

filepath = os.path.dirname(os.path.realpath(__file__)) + '\\data\\' + file_to_save

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

print(data.head())

close_prices = data['Close'].values
close_prices = close_prices.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1)) # normalize
scaled_data = scaler.fit_transform(close_prices)

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])  # Previous time_steps values
        y.append(data[i, 0])               # Current value
    return np.array(X), np.array(y)

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# time steps = number of previous days to consider (adjustable)
time_steps = 30

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(time_steps, 1)))
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
model.add(tf.keras.layers.Dense(units=25))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test = y_test.reshape(-1,1)
y_test_inv = scaler.inverse_transform(y_test)

mse_test = mean_squared_error(y_test_inv, predictions)
print(f"Test MSE: {mse_test}")

rmse_test = np.sqrt(mse_test)
print(f"RMSE: {rmse_test}")

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

last_sequence = test_data[-time_steps:]
future_days = 30 
future_predictions = predict_future(model, last_sequence, future_days)

future_predictions_inv = scaler.inverse_transform(future_predictions)

print(f"Last future prediction (inverse transformed): {future_predictions_inv[-1]}")

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

plt.figure(figsize=(14, 7))

plt.plot(data.index, close_prices, label='Historical Prices', color='blue')

test_dates = data.index[train_size + time_steps:]
plt.plot(test_dates, predictions, label='Predicted Prices', color='orange')

plt.plot(future_dates, future_predictions_inv, label='Future Predictions', color='red', linestyle='--')

plt.title('Stock Price Prediction ($%s)'%ticker, fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
