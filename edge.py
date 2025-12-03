import matplotlib
matplotlib.use("TkAgg")
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from collections import deque
from tensorflow.keras.models import load_model

# ---------- MQTT Setup ----------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"

# ---------- Load Trained LSTM Model + Scaler ----------
print("Loading model and scaler...")
model = load_model("ecg_lstm_model.h5", compile=False)
model.compile(loss="mse", optimizer="adam")
sequence_length = model.input_shape[1]  # e.g., 50

# Load the scaler parameters
scaler_mean = np.load("scaler_mean.npy")[0]
scaler_scale = np.load("scaler_scale.npy")[0]

print(f"Model loaded! Sequence length: {sequence_length}")
print(f"Scaler - Mean: {scaler_mean:.4f}, Scale: {scaler_scale:.4f}")

# ---------- Data Buffer ----------
BUFFER_SIZE = 250
ecg_signal = deque(maxlen=BUFFER_SIZE)
pred_signal = []
data_lock = threading.Lock()
pred_lock = threading.Lock()

# Initialize buffer with zeros
for _ in range(BUFFER_SIZE):
    ecg_signal.append(0.0)

# ---------- Plot Setup ----------
fig, ax = plt.subplots(figsize=(12, 4))
line_real, = ax.plot([], [], color="red", label="Real ECG (200 samples)", 
                      linewidth=2, marker='o', markersize=3)
line_pred, = ax.plot([], [], color="blue", label="Enhanced ECG (LSTM)", 
                     linewidth=1.5, alpha=0.8)
ax.set_xlabel("Sample index")
ax.set_ylabel("Amplitude (mV)")
ax.set_title("Real-time ECG Enhancement with LSTM Prediction")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, BUFFER_SIZE * 2)
ax.set_ylim(-1, 1)
ax.legend(loc='upper right')

# ---------- MQTT Callbacks ----------
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    try:
        message = msg.payload.decode()
        parts = message.strip().split(",")
        values = [float(v) for v in parts[1:]]  # Skip timestamp
        
        with data_lock:
            ecg_signal.extend(values)
    except Exception as e:
        print(f"Error processing message: {e}")

# ---------- Fast Batch Signal Enhancement with Normalization ----------
def enhance_signal_batch(signal_data):
    """
    Batch predict all intermediate points with proper normalization
    """
    if len(signal_data) < sequence_length + 1:
        return signal_data
    
    signal_array = np.array(signal_data)
    
    # NORMALIZE input data using saved scaler
    signal_normalized = (signal_array - scaler_mean) / scaler_scale
    
    # Prepare prediction windows
    num_predictions = len(signal_normalized) - sequence_length
    
    if num_predictions <= 0:
        return signal_data
    
    # Create all windows at once
    windows = np.array([
        signal_normalized[i:i + sequence_length] 
        for i in range(num_predictions)
    ])
    
    # Reshape for LSTM: (batch, timesteps, features)
    windows = windows.reshape(num_predictions, sequence_length, 1)
    
    # BATCH PREDICT (normalized space)
    predictions_normalized = model.predict(windows, verbose=0).flatten()
    
    # DENORMALIZE predictions back to original scale
    predictions = predictions_normalized * scaler_scale + scaler_mean
    
    # Build enhanced signal: interleave real and predicted points
    enhanced = []
    
    # First sequence_length points (no predictions yet)
    for i in range(sequence_length):
        enhanced.append(signal_data[i])
    
    # Interleave: predicted point, then real point
    for i in range(sequence_length, len(signal_data)):
        pred_idx = i - sequence_length
        if pred_idx < len(predictions):
            enhanced.append(predictions[pred_idx])  # Predicted point
        enhanced.append(signal_data[i])  # Real point
    
    return enhanced

# ---------- Animation Function ----------
frame_count = 0
def update_plot(frame):
    global frame_count, pred_signal
    
    with data_lock:
        real_data = list(ecg_signal)
    
    # Update predictions every 3 frames to reduce CPU load
    frame_count += 1
    if frame_count % 3 == 0:
        if len(real_data) >= sequence_length + 1:
            enhanced_data = enhance_signal_batch(real_data)
            with pred_lock:
                pred_signal = enhanced_data
        else:
            with pred_lock:
                pred_signal = real_data
    
    # Get current predicted signal
    with pred_lock:
        current_pred = pred_signal if pred_signal else real_data
    
    # Update plot lines
    real_x = list(range(len(real_data)))
    line_real.set_data(real_x, real_data)
    
    enhanced_x = range(len(current_pred))
    line_pred.set_data(enhanced_x, current_pred)
    
    # Auto-adjust axes
    if current_pred and real_data:
        max_x = max(len(current_pred), len(real_data))
        ax.set_xlim(0, max_x)
        
        all_data = real_data + current_pred
        if all_data:
            y_min, y_max = min(all_data), max(all_data)
            margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 0.5
            ax.set_ylim(y_min - margin, y_max + margin)
    
    return line_real, line_pred

# ---------- Start MQTT Client ----------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("✓ Connected! Waiting for ECG data...")
except Exception as e:
    print(f"✗ Error connecting to MQTT broker: {e}")
    exit(1)

# Start animation
print("Starting real-time visualization...")
ani = animation.FuncAnimation(
    fig, 
    update_plot, 
    interval=50,  # 20 FPS
    blit=True, 
    cache_frame_data=False
)

try:
    plt.show()
except KeyboardInterrupt:
    print("\\n\\nShutting down...")
finally:
    client.loop_stop()
    client.disconnect()
    print("✓ Disconnected from MQTT broker")
