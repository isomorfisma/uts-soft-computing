import streamlit as st
import matplotlib.pyplot as plt
from core.fis_manual import get_manual_fis, predict_manual
from core.fis_ga import get_ga_tuned_fis, predict_ga
from core.ann_model import predict_ann, get_ann_loss_curve

st.set_page_config(page_title="Analisis Strategi Credit Scoring", layout="wide")

# --- CUSTOM CSS UNTUK STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ff0000;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("Sistem Analisis Kelayakan Kredit")
st.write("Perbandingan Metode Fuzzy Inference System (Manual & GA) dan Artificial Neural Networks")
st.markdown("---")

# --- SIDEBAR INPUT ---
st.sidebar.header("Data Masukan Nasabah")
val_pendapatan = st.sidebar.number_input("Pendapatan Bulanan (Juta Rp)", min_value=0.0, max_value=50.0, value=7.0, step=0.5)
val_utang = st.sidebar.number_input("Persentase Rasio Utang (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Pilih Metode Analisis:", ["Manual FIS (Pakar)", "GA-Tuned FIS", "ANN Optimasi"])

# --- FUNGSI PLOTTING MANUAL ---
def plot_mf(var, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    for label in var.terms:
        ax.plot(var.universe, var[label].mf, label=label, linewidth=2)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig

# --- LOGIKA PENJELASAN ALGORITMA ---
explanations = {
    "Manual FIS (Pakar)": {
        "deskripsi": "Metode ini sepenuhnya didasarkan pada intuisi manusia atau pakar keuangan.",
        "asumsi": "Manusia dianggap mampu menentukan ambang batas nilai yang pasti. Misalnya, menetapkan secara kaku bahwa rasio utang di atas 30% adalah 'Bahaya'.",
        "proses": "Input dikonversi menjadi nilai fuzzy, lalu diproses menggunakan logika IF-THEN yang disusun manual oleh kelompok sebelum menghasilkan keputusan akhir."
    },
    "GA-Tuned FIS": {
        "deskripsi": "Metode ini menggunakan Algoritma Genetika (GA) untuk memperbaiki parameter sistem fuzzy.",
        "asumsi": "Batas kurva yang dibuat manusia mungkin tidak optimal. Sistem berasumsi bahwa ada konfigurasi parameter yang lebih akurat jika disesuaikan dengan data riil.",
        "proses": "Terjadi simulasi evolusi (seleksi, persilangan, dan mutasi). GA mencari kombinasi titik kurva yang menghasilkan tingkat kesalahan (error) paling rendah terhadap data historis."
    },
    "ANN Optimasi": {
        "deskripsi": "Metode Jaringan Saraf Tiruan (ANN) bekerja menyerupai cara kerja saraf biologis.",
        "asumsi": "Sistem tidak memerlukan aturan (rules) kaku. Diasumsikan bahwa hubungan antara pendapatan dan risiko dapat ditemukan melalui pola matematika murni dari data.",
        "proses": "Mesin melakukan 'training' dengan menyesuaikan bobot (weights) pada tiap lapisan saraf hingga model mampu memprediksi kelayakan dengan akurasi tinggi secara mandiri."
    }
}

# --- MAIN LAYOUT ---
tab_kalkulator, tab_teori = st.tabs(["Kalkulator Keputusan", "Detail Metodologi"])

with tab_kalkulator:
    col_res, col_vis = st.columns([1, 1.5])
    
    with col_res:
        st.subheader("Hasil Keputusan")
        current_exp = explanations[mode]
        st.write(f"**Analisis:** {current_exp['deskripsi']}")
        
        container = st.container()
        if mode == "Manual FIS (Pakar)":
            hasil = predict_manual(val_pendapatan, val_utang)
            label_score = "Skor Kelayakan Manual"
        elif mode == "GA-Tuned FIS":
            with st.spinner("Mengoptimasi parameter kurva..."):
                hasil = predict_ga(val_pendapatan, val_utang)
            label_score = "Skor Kelayakan GA"
        else:
            hasil = predict_ann(val_pendapatan, val_utang)
            label_score = "Skor Kelayakan ANN"
            
        container.metric(label=label_score, value=f"{hasil:.2f}%")
        if hasil >= 50:
            container.success("REKOMENDASI: TERIMA PINJAMAN")
        else:
            container.error("REKOMENDASI: TOLAK PINJAMAN")

    with col_vis:
        st.subheader("Visualisasi Logika Mesin")
        if mode != "ANN Optimasi":
            if mode == "Manual FIS (Pakar)":
                _, var_pend, var_ut, _ = get_manual_fis()
            else:
                _, var_pend, var_ut, _, best_params = get_ga_tuned_fis()
                st.caption(f"Parameter Optimal GA: {best_params}")
            
            st.pyplot(plot_mf(var_pend, "Kurva Membership: Pendapatan"))
            st.pyplot(plot_mf(var_ut, "Kurva Membership: Rasio Utang"))
        else:
            st.write("Model ANN bersifat Black-Box (tidak menggunakan kurva membership).")
            st.write("Berikut adalah progres penurunan error selama fase pembelajaran:")
            st.pyplot(get_ann_loss_curve())

with tab_teori:
    st.subheader("Transparansi Algoritma")
    for key, val in explanations.items():
        with st.expander(f"Bagaimana {key} Bekerja?"):
            st.write(f"**Prinsip Dasar:** {val['deskripsi']}")
            st.write(f"**Asumsi Dasar:** {val['asumsi']}")
            st.write(f"**Proses Belakang Layar:** {val['proses']}")