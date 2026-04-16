import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygad
import pandas as pd
import os

# 1. Load Dataset untuk Evaluasi Fitness (Cari error terkecil)
# Pastikan path relatifnya benar menuju data_dummy.csv
data_path = os.path.join(os.path.dirname(__file__), '../dataset/data_dummy.csv')
df = pd.read_csv(data_path)

# Ambil sampel 50 data aja biar GA bisa jalan realtime di Streamlit tanpa lag
X_pendapatan = df['Pendapatan'].values[:50]
X_utang = df['Rasio_Utang'].values[:50]
y_true = df['Kelayakan'].values[:50]

# 2. Template FIS yang Parameternya Bisa Diubah (Mutasi)
def create_fis_with_params(params):
    pendapatan = ctrl.Antecedent(np.arange(0, 21, 1), 'pendapatan')
    rasio_utang = ctrl.Antecedent(np.arange(0, 101, 1), 'rasio_utang')
    kelayakan = ctrl.Consequent(np.arange(0, 101, 1), 'kelayakan')

    # Genes dari GA: [batas_rendah, puncak_menengah, batas_tinggi, batas_utang_bahaya]
    p1, p2, p3, p4 = params
    
    # Pengaman biar kurva ga tumpang tindih aneh-aneh akibat mutasi ekstrem
    p1 = max(2, min(p1, 8))     # Batas Rendah (Manusia: 7)
    p2 = max(p1+1, min(p2, 14)) # Puncak Menengah (Manusia: 10)
    p3 = max(p2+1, min(p3, 18)) # Batas Tinggi (Manusia: 12)
    p4 = max(20, min(p4, 60))   # Batas Utang Bahaya (Manusia: 30)

    # Membentuk kurva baru berdasarkan kromosom GA
    pendapatan['rendah'] = fuzz.trapmf(pendapatan.universe, [0, 0, p1-2, p1])
    pendapatan['menengah'] = fuzz.trimf(pendapatan.universe, [p1-1, p2, p3+1])
    pendapatan['tinggi'] = fuzz.trapmf(pendapatan.universe, [p3, p3+2, 20, 20])

    rasio_utang['aman'] = fuzz.trapmf(rasio_utang.universe, [0, 0, p4-10, p4])
    rasio_utang['bahaya'] = fuzz.trapmf(rasio_utang.universe, [p4-5, p4+10, 100, 100])

    kelayakan['tolak'] = fuzz.trimf(kelayakan.universe, [0, 0, 50])
    kelayakan['terima'] = fuzz.trimf(kelayakan.universe, [40, 100, 100])

    # Rules tetap sama
    rule1 = ctrl.Rule(pendapatan['rendah'] | rasio_utang['bahaya'], kelayakan['tolak'])
    rule2 = ctrl.Rule(pendapatan['tinggi'] & rasio_utang['aman'], kelayakan['terima'])
    rule3 = ctrl.Rule(pendapatan['menengah'] & rasio_utang['aman'], kelayakan['terima'])
    rule4 = ctrl.Rule(pendapatan['menengah'] & rasio_utang['bahaya'], kelayakan['tolak'])

    scoring_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(scoring_ctrl), pendapatan, rasio_utang, kelayakan

# 3. Fungsi Fitness (Menghitung Mean Squared Error / MSE)
def fitness_func(ga_instance, solution, solution_idx):
    sim, _, _, _ = create_fis_with_params(solution)
    error = 0
    for i in range(len(X_pendapatan)):
        sim.input['pendapatan'] = X_pendapatan[i]
        sim.input['rasio_utang'] = X_utang[i]
        try:
            sim.compute()
            pred = sim.output['kelayakan']
            error += (pred - y_true[i])**2
        except:
            error += 10000 # Penalti berat kalau sistem gagal (rule bolong)
    
    mse = error / len(X_pendapatan)
    # Fitness = 1 / MSE. Makin kecil error, makin besar fitness.
    return 1.0 / (mse + 1e-6) 

# 4. Fungsi Utama untuk Dijalankan
def get_ga_tuned_fis():
    # Inisialisasi GA
    ga_instance = pygad.GA(
        num_generations=15,       # Berapa kali mutasi/evolusi
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=10,           # Jumlah kromosom per generasi
        num_genes=4,
        # Batas ruang pencarian untuk [p1, p2, p3, p4]
        gene_space=[{'low': 3, 'high': 8}, {'low': 8, 'high': 14}, {'low': 11, 'high': 18}, {'low': 20, 'high': 60}],
        mutation_probability=0.2  # Ablation study: Ubah ini nanti di laporan
    )
    
    ga_instance.run()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    
    sim, var_pendapatan, var_utang, var_kelayakan = create_fis_with_params(best_solution)
    return sim, var_pendapatan, var_utang, var_kelayakan, best_solution

def predict_ga(val_pendapatan, val_utang):
    sim, _, _, _, _ = get_ga_tuned_fis()
    sim.input['pendapatan'] = val_pendapatan
    sim.input['rasio_utang'] = val_utang
    sim.compute()
    return sim.output['kelayakan']