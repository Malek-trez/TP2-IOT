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

# ---------- Paramètres d'affichage ----------
SAMPLE_RATE = 200  # 200 échantillons par seconde
duree_periode = 1.0  # 1 seconde par cycle
periodes_a_afficher = 5  # Afficher 5 cycles
BUFFER_SIZE = int(SAMPLE_RATE * duree_periode * periodes_a_afficher)  # 200 * 1 * 5 = 1000

# ---------- Échelle d'amplitude constante ----------
AMPLITUDE_MIN = -0.8  # Minimum d'amplitude en mV
AMPLITUDE_MAX = 1.8   # Maximum d'amplitude en mV

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
print(f"Buffer size: {BUFFER_SIZE} samples ({periodes_a_afficher} cycles)")
print(f"Amplitude scale fixed: [{AMPLITUDE_MIN}, {AMPLITUDE_MAX}] mV")

# ---------- Data Buffer ----------
ecg_signal = deque(maxlen=BUFFER_SIZE)
pred_signal = []
data_lock = threading.Lock()
pred_lock = threading.Lock()

# Initialize buffer with zeros
for _ in range(BUFFER_SIZE):
    ecg_signal.append(0.0)

# ---------- Time array for x-axis ----------
# Créer un tableau de temps en secondes pour les 5 cycles
t_buffer = np.linspace(0, duree_periode * periodes_a_afficher, BUFFER_SIZE)

# ---------- Plot Setup ----------
fig, ax = plt.subplots(figsize=(12, 4))
line_real, = ax.plot([], [], color="red", label="Real ECG (200 samples/sec)", 
                      linewidth=2, marker='o', markersize=3)
line_pred, = ax.plot([], [], color="blue", label="Enhanced ECG (LSTM)", 
                     linewidth=1.5, alpha=0.8)

# Ajouter des lignes verticales pour marquer les cycles
for i in range(periodes_a_afficher + 1):
    ax.axvline(x=i * duree_periode, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

# Ajouter une ligne horizontale pour le zéro
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

# Ajouter une grille pour faciliter la lecture
ax.grid(True, alpha=0.2)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude (mV)")
ax.set_title(f"Real-time ECG Enhancement with LSTM Prediction ({periodes_a_afficher} cycles = {periodes_a_afficher} seconds)")
ax.set_xlim(0, duree_periode * periodes_a_afficher)  # De 0 à 5 secondes
ax.set_ylim(AMPLITUDE_MIN, AMPLITUDE_MAX)  # Échelle d'amplitude constante
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
    
    # Calculer le temps pour les données prédites
    # Les données prédites ont environ 2x plus de points que les données réelles
    if current_pred:
        t_pred = np.linspace(0, duree_periode * periodes_a_afficher, len(current_pred))
    else:
        t_pred = np.array([])
    
    # Update plot lines
    line_real.set_data(t_buffer[:len(real_data)], real_data)
    
    if len(t_pred) > 0 and len(current_pred) > 0:
        line_pred.set_data(t_pred, current_pred)
    
    # L'échelle Y reste constante - pas d'ajustement automatique
    # L'échelle X reste également constante (0 à 5 secondes)
    
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
    print(f"Displaying {periodes_a_afficher} cycles ({duree_periode * periodes_a_afficher} seconds)")
    print(f"Amplitude scale fixed at [{AMPLITUDE_MIN}, {AMPLITUDE_MAX}] mV")
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
    print("\n\nShutting down...")
finally:
    client.loop_stop()
    client.disconnect()
    print("✓ Disconnected from MQTT broker")