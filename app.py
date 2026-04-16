import streamlit as st
import matplotlib.pyplot as plt
from core.fis_manual import get_manual_fis, predict_manual
from core.fis_ga import get_ga_tuned_fis, predict_ga
from core.ann_model import predict_ann, get_ann_loss_curve

st.set_page_config(page_title="Credit Scoring UTS", layout="wide")

st.title("🏦 Credit Scoring System")

st.sidebar.header("Input Nasabah")
val_pendapatan = st.sidebar.slider("Pendapatan (Juta Rp)", 0.0, 20.0, 5.0)
val_utang = st.sidebar.slider("Rasio Utang (%)", 0.0, 100.0, 30.0)

# INI BAGIAN YANG LO SKIP TADI
mode = st.sidebar.radio("Pilih Engine:", ("Manual FIS (Pakar)", "GA-Tuned FIS", "ANN Optimasi"))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hasil Analisis")
    if mode == "Manual FIS (Pakar)":
        hasil = predict_manual(val_pendapatan, val_utang)
        st.metric(label="Skor Kelayakan (Manual)", value=f"{hasil:.2f}%")
        if hasil >= 50:
            st.success("Keputusan: DITERIMA")
        else:
            st.error("Keputusan: DITOLAK")

    elif mode == "GA-Tuned FIS":
        with st.spinner("Mesin sedang melakukan proses evolusi GA... (sekitar 3-5 detik)"):
            hasil_ga = predict_ga(val_pendapatan, val_utang)
            st.metric(label="Skor Kelayakan (Tuned by GA)", value=f"{hasil_ga:.2f}%")
            if hasil_ga >= 50:
                st.success("Keputusan: DITERIMA")
            else:
                st.error("Keputusan: DITOLAK")

    elif mode == "ANN Optimasi":
        hasil_ann = predict_ann(val_pendapatan, val_utang)
        st.metric(label="Skor Kelayakan (ANN)", value=f"{hasil_ann:.2f}%")
        if hasil_ann >= 50:
            st.success("Keputusan: DITERIMA")
        else:
            st.error("Keputusan: DITOLAK")

with col2:
    st.subheader("Visualisasi Kurva Membership")
    if mode == "Manual FIS (Pakar)":
        # NAMA FUNGSI UDAH GUE BENERIN JADI get_manual_fis
        _, var_pendapatan, _, _ = get_manual_fis() 
        fig, ax = plt.subplots()
        var_pendapatan.view(sim=None, ax=ax)
        st.pyplot(fig)
        
    elif mode == "GA-Tuned FIS":
        with st.spinner("Merender kurva hasil evolusi..."):
            _, var_pendapatan, _, _, best_params = get_ga_tuned_fis()
            fig, ax = plt.subplots()
            var_pendapatan.view(sim=None, ax=ax)
            st.pyplot(fig)
            
            # Tampilkan parameter kromosom terbaik biar dosen lo liat buktinya
            st.write("**Kromosom Terbaik yang Ditemukan GA:**")
            st.code(f"Batas Pendapatan Rendah: {best_params[0]:.2f}\nPuncak Pend. Menengah: {best_params[1]:.2f}\nBatas Pendapatan Tinggi: {best_params[2]:.2f}\nBatas Utang Bahaya: {best_params[3]:.2f}")
            
    elif mode == "ANN Optimasi":
        st.info("ANN tidak memiliki kurva Membership Function (Sistem Black-Box).")
        st.write("Sebagai gantinya, ini adalah kurva penurunan Error (Loss) selama ANN mempelajari dataset historis:")
        fig = get_ann_loss_curve()
        st.pyplot(fig)