# Version avec buffer pour afficher plusieurs périodes
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import time
from collections import deque

# Configuration MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"

def ecg_synthetique(t):
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
    )


# Initialiser le client MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

# Paramètres
points_par_periode = 500
nb_echantillons=250
duree_periode = 1.0
t_current = np.linspace(0, duree_periode, points_par_periode)
periode_compteur = 0

try:
    while True:
        # Générer la période ECG actuelle
        current_real = ecg_synthetique(t_current)
        bruit = 0.05 * np.random.normal(0, 0.05, len(signal))
        current_noisy = current_real + bruit
        
        # Échantillonnage: 5 points par période
        indices_echantillons = np.linspace(0, points_par_periode-1, nb_echantillons, dtype=int)
        t_echantillons = t_current[indices_echantillons]
        echantillons = current_noisy[indices_echantillons]
        
        # Envoyer via MQTT
        timestamp = time.time()
        mqtt_data = f"{timestamp},{','.join([f'{val:.4f}' for val in echantillons])}"
        mqtt_client.publish(MQTT_TOPIC, mqtt_data)
        
        # Informations console
        print(f"\n=== Période {periode_compteur + 1} ===")
        print(f"MQTT: {mqtt_data}")
        
        periode_compteur += 1
        time.sleep(duree_periode)
        
except KeyboardInterrupt:
    print("\nArrêt du programme...")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    plt.ioff()
    plt.show()
