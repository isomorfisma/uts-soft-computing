import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Load Dataset
data_path = os.path.join(os.path.dirname(__file__), '../dataset/data_dummy.csv')
df = pd.read_csv(data_path)

X = df[['Pendapatan', 'Rasio_Utang']].values
y = df['Kelayakan'].values

# 2. Setup dan Training Model ANN (Multi-Layer Perceptron)
# Kita pakai 2 hidden layer (10 neuron dan 5 neuron)
# max_iter=2000 biar dia konvergen (belajar sampai tuntas)
ann_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=2000, random_state=42, early_stopping=False)
ann_model.fit(X, y)

def predict_ann(val_pendapatan, val_utang):
    # ANN scikit-learn butuh input bentuk 2D array
    input_data = np.array([[val_pendapatan, val_utang]])
    prediksi = ann_model.predict(input_data)[0]
    
    # Pastikan hasil tidak keluar jalur (0-100%)
    return max(0.0, min(prediksi, 100.0))

def get_ann_loss_curve():
    # Mengambil kurva error (loss) selama proses training buat ditampilin di GUI
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(ann_model.loss_curve_, color='purple')
    ax.set_title("Kurva Pembelajaran ANN (Loss Curve)")
    ax.set_xlabel("Iterasi (Epochs)")
    ax.set_ylabel("Error (Loss)")
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig