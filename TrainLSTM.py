import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# 1. Génération d’un signal ECG simplifié
def ecg_cycle(t):
    return (
    0.1 * np.sin(2 * np.pi * t * 1) +
    -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
    1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +

    10

    -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
    0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
    )
t = np.linspace(0, 1, 500)
signal = np.tile(ecg_cycle(t), 10) # 10 cycles
# 2. Préparation des séquences
sequence_length = 50
X, y = [], []
for i in range(len(signal) - sequence_length):
    X.append(signal[i:i+sequence_length])
    y.append(signal[i+sequence_length])
X = np.array(X)
y = np.array(y)
# Reshape en (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
# 3. Modèle LSTM
model = Sequential([
                    LSTM(64, input_shape=(sequence_length, 1)),
                    Dense(1)
                    ])
model.compile(optimizer='adam', loss='mse')
# 4. Entraînement
model.fit(X, y, epochs=10, batch_size=32)

model.save("ecg_lstm_model.h5")