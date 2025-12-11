# Version avec buffer pour afficher plusieurs périodes
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import time
from matplotlib.widgets import Button
import threading

# Configuration MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"

# Initialiser le client MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

def ecg_normal(t):
    """Fonction pour générer un cycle ECG normal."""
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
    )

def ecg_extrasystole(t):
    """Fonction pour générer un cycle ECG avec extrasystole à CHAQUE battement."""
    # Extrasystole ventriculaire prématurée - se produit à chaque battement
    return (
        0.15 * np.sin(2 * np.pi * t * 1.3) +  # Fréquence plus rapide
        -0.25 * np.exp(-((t - 0.15) ** 2) / 0.001) +  # P wave modifiée
        1.5 * np.exp(-((t - 0.25) ** 2) / 0.0003) +   # QRS plus large et plus haut
        -0.35 * np.exp(-((t - 0.35) ** 2) / 0.001) +  # S wave modifiée
        0.4 * np.exp(-((t - 0.6) ** 2) / 0.015)       # T wave décalée
    )

# Variables globales pour le mode ECG
ecg_mode = "normal"  # "normal" ou "extrasystole"
beat_counter = 0
mode_lock = threading.Lock()

# Paramètres
points_par_periode = 500
duree_periode = 1.0
periodes_a_afficher = 10

# Initialiser le graphique avec un layout pour le bouton
plt.ion()
fig = plt.figure(figsize=(14, 9))

# Créer une grille pour organiser les subplots et le bouton
gs = plt.GridSpec(3, 1, height_ratios=[3, 2, 0.5])

# Graphique 1: Vue d'ensemble
ax1 = plt.subplot(gs[0])
t_buffer = np.linspace(0, duree_periode * periodes_a_afficher, points_par_periode * periodes_a_afficher)
real_buffer = np.zeros(points_par_periode * periodes_a_afficher)
noisy_buffer = np.zeros(points_par_periode * periodes_a_afficher)

line_real_all, = ax1.plot(t_buffer, real_buffer, 'b-', label='ECG réel', alpha=0.6, linewidth=1)
line_noisy_all, = ax1.plot(t_buffer, noisy_buffer, 'r-', label='ECG bruité', alpha=0.8, linewidth=1)
line_samples_all, = ax1.plot([], [], 'go', label='Points échantillonnés', markersize=6, alpha=0.6)

ax1.set_title("ECG - Vue d'ensemble (Mode: Normal)")
ax1.set_xlabel("Temps (s)")
ax1.set_ylabel("Amplitude (mV)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xlim(0, duree_periode * periodes_a_afficher)
ax1.set_ylim(-0.6, 2.0)

# Graphique 2: Vue détaillée
ax2 = plt.subplot(gs[1])
t_current = np.linspace(0, duree_periode, points_par_periode)
line_real_current, = ax2.plot(t_current, np.zeros_like(t_current), 'b-', label='ECG réel', linewidth=2)
line_noisy_current, = ax2.plot(t_current, np.zeros_like(t_current), 'r-', label='ECG bruité', alpha=0.7, linewidth=2)
line_samples_current, = ax2.plot([], [], 'go', label='Points échantillonnés', markersize=10, markeredgewidth=2)

ax2.set_title("Dernière période ECG - Vue détaillée (Mode: Normal)")
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel("Amplitude (mV)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_xlim(0, duree_periode)
ax2.set_ylim(-0.6, 2.0)

# Zone pour le bouton
ax_button = plt.subplot(gs[2])
ax_button.axis('off')

# Créer les boutons
button_normal_ax = plt.axes([0.3, 0.02, 0.2, 0.07])
button_extrasystole_ax = plt.axes([0.55, 0.02, 0.2, 0.07])

button_normal = Button(button_normal_ax, 'Mode Normal', color='lightblue', hovercolor='lightgreen')
button_extrasystole = Button(button_extrasystole_ax, 'Mode Extrasystole', color='lightcoral', hovercolor='orange')

# Texte d'information
info_ax = plt.axes([0.02, 0.02, 0.25, 0.07])
info_ax.axis('off')
info_text = info_ax.text(0, 0.5, 'Mode: Normal', fontsize=12, fontweight='bold', color='darkblue')

# Fonctions pour changer le mode
def set_normal_mode(event):
    global ecg_mode
    with mode_lock:
        ecg_mode = "normal"
        button_normal.color = 'lightgreen'
        button_extrasystole.color = 'lightcoral'
        info_text.set_text('Mode: Normal')
        info_text.set_color('darkblue')
        print("\n✓ Mode changé : Normal (rythme sinusal régulier)")

def set_extrasystole_mode(event):
    global ecg_mode
    with mode_lock:
        ecg_mode = "extrasystole"
        button_normal.color = 'lightblue'
        button_extrasystole.color = 'red'
        info_text.set_text('Mode: Extrasystole')
        info_text.set_color('darkred')
        print("\n✓ Mode changé : Extrasystole (PVC à chaque battement)")

# Connecter les boutons aux fonctions
button_normal.on_clicked(set_normal_mode)
button_extrasystole.on_clicked(set_extrasystole_mode)

plt.tight_layout()

periode_compteur = 0
sample_positions = []

def update_title():
    """Mettre à jour le titre avec le mode actuel."""
    if ecg_mode == "normal":
        ax1.set_title("ECG - Vue d'ensemble (Mode: Normal)")
        ax2.set_title("Dernière période ECG - Vue détaillée (Mode: Normal)")
    else:
        ax1.set_title("ECG - Vue d'ensemble (Mode: Extrasystole)")
        ax2.set_title("Dernière période ECG - Vue détaillée (Mode: Extrasystole)")

try:
    while True:
        # Générer la période ECG actuelle selon le mode
        if ecg_mode == "normal":
            current_real = ecg_normal(t_current)
            beat_type = "Normal"
        else:
            current_real = ecg_extrasystole(t_current)
            beat_type = "Extrasystole"
        
        # Ajouter du bruit
        if ecg_mode == "normal":
            bruit = 0.02 * np.random.normal(size=current_real.shape)
        else:
            bruit = 0.01 * np.random.normal(size=current_real.shape)  # Moins de bruit pour extrasystole
        
        current_noisy = current_real + bruit
        
        # Échantillonnage
        indices_echantillons = np.linspace(0, points_par_periode-1, 200, dtype=int)
        t_echantillons = t_current[indices_echantillons]
        echantillons = current_noisy[indices_echantillons]
        
        # Envoyer via MQTT
        timestamp = time.time()
        mqtt_data = f"{timestamp},{','.join([f'{val:.4f}' for val in echantillons])}"
        mqtt_client.publish(MQTT_TOPIC, mqtt_data)
        
        # Mettre à jour les buffers
        start_idx = (periode_compteur % periodes_a_afficher) * points_par_periode
        real_buffer[start_idx:start_idx+points_par_periode] = current_real
        noisy_buffer[start_idx:start_idx+points_par_periode] = current_noisy
        
        # Calculer les positions des échantillons dans le buffer
        sample_times = t_echantillons + (periode_compteur % periodes_a_afficher) * duree_periode
        sample_positions = list(zip(sample_times, echantillons))
        
        # Mettre à jour les graphiques
        # Graphique d'ensemble
        line_real_all.set_ydata(real_buffer)
        line_noisy_all.set_ydata(noisy_buffer)
        
        if sample_positions:
            sample_times_all, sample_vals_all = zip(*sample_positions)
            line_samples_all.set_data(sample_times_all, sample_vals_all)
        
        # Graphique détaillé (dernière période)
        line_real_current.set_ydata(current_real)
        line_noisy_current.set_ydata(current_noisy)
        line_samples_current.set_data(t_echantillons, echantillons)
        
        # Mettre à jour les titres
        update_title()
        
        # Redessiner
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Informations console (afficher moins fréquemment)
        if periode_compteur % 5 == 0:  
            print(f"\n=== Battement {periode_compteur + 1} ===")
            print(f"Mode: {ecg_mode.upper()}")
            print(f"Type de battement: {beat_type}")
            print(f"Points envoyés via MQTT: {len(echantillons)}")
            if ecg_mode == "extrasystole":
                print("⚠️  EXTRASYSTOLE VENTRICULAIRE PRÉMATURÉE (PVC)")
        
        periode_compteur += 1
        time.sleep(duree_periode)
        
except KeyboardInterrupt:
    print("\nArrêt du programme...")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    plt.ioff()
    plt.show()