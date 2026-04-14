import pandas as pd
import numpy as np

# Set seed biar datanya konsisten kalau di-run ulang
np.random.seed(42)

jumlah_data = 500

# Generate input
pendapatan = np.random.uniform(2, 20, jumlah_data)  # Gaji 2 - 20 Juta
rasio_utang = np.random.uniform(0, 80, jumlah_data)  # Utang 0 - 80% dari gaji

# Logika realita (ground truth) buat nentuin kelayakan kredit
# Makin tinggi gaji dan makin rendah utang = makin layak
skor_dasar = (pendapatan / 20 * 60) + ((80 - rasio_utang) / 80 * 40)

# Noise
noise = np.random.normal(0, 5, jumlah_data)
kelayakan = np.clip(skor_dasar + noise, 0, 100)  # Batasi 0-100%

df = pd.DataFrame({
    'Pendapatan': np.round(pendapatan, 2),
    'Rasio_Utang': np.round(rasio_utang, 2),
    'Kelayakan': np.round(kelayakan, 2)
})

df.to_csv('dataset/data_dummy.csv', index=False)
print("Dataset berhasil dibuat di dataset/data_dummy.csv!")