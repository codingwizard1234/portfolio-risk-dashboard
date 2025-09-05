import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(series, time_steps=10):
    X, y = [], []
    for i in range(len(series)-time_steps):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps])
    return np.array(X), np.array(y)

def train_lstm_volatility(returns, time_steps=10, epochs=20):
    vol = returns.rolling(window=5).std().dropna().values.reshape(-1,1)
    scaler = MinMaxScaler()
    vol_scaled = scaler.fit_transform(vol)

    X, y = prepare_lstm_data(vol_scaled, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(time_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    last_sequence = vol_scaled[-time_steps:].reshape((1, time_steps, 1))
    pred = model.predict(last_sequence)
    return scaler.inverse_transform(pred)[0][0]
