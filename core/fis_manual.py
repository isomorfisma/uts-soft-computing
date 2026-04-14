import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_manual_fis():
    # 1. Definisi Variabel (Antecedent = Input, Consequent = Output)
    pendapatan = ctrl.Antecedent(np.arange(0, 21, 1), 'pendapatan')
    rasio_utang = ctrl.Antecedent(np.arange(0, 101, 1), 'rasio_utang')
    kelayakan = ctrl.Consequent(np.arange(0, 101, 1), 'kelayakan')

    # 2. Membership Function (Logika Pakar Manusia)
    # Gaji: Rendah (<7), Menengah (5-15), Tinggi (>12)
    pendapatan['rendah'] = fuzz.trapmf(pendapatan.universe, [0, 0, 3, 7])
    pendapatan['menengah'] = fuzz.trimf(pendapatan.universe, [5, 10, 15])
    pendapatan['tinggi'] = fuzz.trapmf(pendapatan.universe, [12, 17, 20, 20])

    # Utang: Aman (<40%), Bahaya (>30%)
    rasio_utang['aman'] = fuzz.trapmf(rasio_utang.universe, [0, 0, 20, 40])
    rasio_utang['bahaya'] = fuzz.trapmf(rasio_utang.universe, [30, 60, 100, 100])

    # Kelayakan: Tolak (<50%), Terima (>40%)
    kelayakan['tolak'] = fuzz.trimf(kelayakan.universe, [0, 0, 50])
    kelayakan['terima'] = fuzz.trimf(kelayakan.universe, [40, 100, 100])

    # 3. Rules Pakar
    rule1 = ctrl.Rule(pendapatan['rendah'] | rasio_utang['bahaya'], kelayakan['tolak'])
    rule2 = ctrl.Rule(pendapatan['tinggi'] & rasio_utang['aman'], kelayakan['terima'])
    rule3 = ctrl.Rule(pendapatan['menengah'] & rasio_utang['aman'], kelayakan['terima'])
    rule4 = ctrl.Rule(pendapatan['menengah'] & rasio_utang['bahaya'], kelayakan['tolak'])

    scoring_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    sim = ctrl.ControlSystemSimulation(scoring_ctrl)
    
    return sim, pendapatan, rasio_utang, kelayakan

def predict_manual(val_pendapatan, val_utang):
    sim, _, _, _ = get_manual_fis()
    sim.input['pendapatan'] = val_pendapatan
    sim.input['rasio_utang'] = val_utang
    sim.compute()
    return sim.output['kelayakan']