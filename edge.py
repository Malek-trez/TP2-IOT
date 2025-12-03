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

# ---------- Load Trained LSTM Model ----------
model = load_model("ecg_lstm_model.h5")  # replace with your saved model path
sequence_length = model.input_shape[1]  # e.g., 50 if trained with 50

# ---------- Data Buffer ----------
PREDICTION_BUFFER = 200  # number of points used for prediction
ecg_signal = deque(maxlen=PREDICTION_BUFFER)
pred_signal = deque(maxlen=PREDICTION_BUFFER)
data_lock = threading.Lock()

# Initialize buffers with zeros
for _ in range(PREDICTION_BUFFER):
    ecg_signal.append(0.0)
    pred_signal.append(0.0)

# ---------- Plot Setup ----------
fig, ax = plt.subplots(figsize=(10, 4))
line_real, = ax.plot([], [], color="red", label="Real ECG")
line_pred, = ax.plot([], [], color="blue", linestyle="--", label="Predicted ECG")
ax.set_xlabel("Sample index")
ax.set_ylabel("Amplitude (mV)")
ax.set_title("Real-time ECG Data with LSTM Prediction")
ax.grid(True)
ax.set_xlim(0, PREDICTION_BUFFER)
ax.set_ylim(-1, 1)
ax.legend()

# ---------- MQTT Callbacks ----------
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        message = msg.payload.decode()
        
        # Parse message: timestamp,val1,val2,...
        parts = message.strip().split(",")
        values = [float(v) for v in parts[1:]]  # skip timestamp
        
        # Update buffer with thread lock
        with data_lock:
            ecg_signal.extend(values)
    except Exception as e:
        print("Error processing message:", e)

# ---------- Animation Function ----------
def update_plot(frame):
    with data_lock:
        real_data = list(ecg_signal)
    
    if len(real_data) >= PREDICTION_BUFFER:
        # Take last PREDICTION_BUFFER points for prediction
        input_seq = np.array(real_data[-PREDICTION_BUFFER:]).reshape(1, PREDICTION_BUFFER, 1)
        next_point = model.predict(input_seq, verbose=0)[0,0]
        
        with data_lock:
            pred_signal.append(next_point)
    
    # Update lines
    with data_lock:
        current_pred = list(pred_signal)
        current_real = list(ecg_signal)
    
    # Ensure both lines have same length for plotting
    length = min(len(current_real), len(current_pred))
    line_real.set_data(range(length), current_real[-length:])
    line_pred.set_data(range(length), current_pred[-length:])
    
    # Auto-adjust Y limits
    all_data = current_real[-length:] + current_pred[-length:]
    y_min, y_max = min(all_data), max(all_data)
    margin = (y_max - y_min) * 0.1 or 0.5
    ax.set_ylim(y_min - margin, y_max + margin)
    
    return line_real, line_pred

# ---------- Start MQTT Client ----------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("Connected to MQTT broker. Starting ECG monitoring...")
except Exception as e:
    print("Error connecting to MQTT broker:", e)

# Start animation
ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=True)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    client.loop_stop()
    client.disconnect()
