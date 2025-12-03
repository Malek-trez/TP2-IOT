import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# 1. Génération d'un signal ECG simplifié (plus de données)
def ecg_cycle(t):
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
    )

# Generate MORE data for better training
t = np.linspace(0, 1, 350)
signal = np.tile(ecg_cycle(t), 100)  # 50 cycles instead of 10!

# Add slight random noise to make it more realistic
noise = np.random.normal(0, 0.05, len(signal))
signal = signal + noise

# 2. NORMALIZE the data (VERY IMPORTANT!)
scaler = StandardScaler()
signal_normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

# 3. Préparation des séquences
sequence_length = 50
X, y = [], []

for i in range(len(signal_normalized) - sequence_length):
    X.append(signal_normalized[i:i+sequence_length])
    y.append(signal_normalized[i+sequence_length])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train/validation
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# 4. IMPROVED LSTM Model
model = Sequential([
    # Bidirectional LSTM looks at sequence both forward and backward
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, 1)),
    Dropout(0.2),  # Prevent overfitting
    
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    
    LSTM(32),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 6. Training with validation
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Will stop early if needed
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 7. Save model AND scaler (IMPORTANT!)
model.save("ecg_lstm_model.h5")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
