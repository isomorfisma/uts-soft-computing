import streamlit as st
import matplotlib.pyplot as plt
from core.fis_manual import get_manual_fis, predict_manual

st.set_page_config(page_title="Credit Scoring UTS", layout="wide")

st.title("🏦 Credit Scoring System")

st.sidebar.header("Input Nasabah")
val_pendapatan = st.sidebar.slider("Pendapatan (Juta Rp)", 0.0, 20.0, 5.0)
val_utang = st.sidebar.slider("Rasio Utang (%)", 0.0, 100.0, 30.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hasil Prediksi Pakar (Manual)")
    try:
        hasil_manual = predict_manual(val_pendapatan, val_utang)
        st.metric("Skor Kelayakan", f"{hasil_manual:.2f}%")
        if hasil_manual >= 50:
            st.success("Keputusan: DITERIMA")
        else:
            st.error("Keputusan: DITOLAK")
    except Exception as e:
        st.error(f"Error pada sistem: Masukkan kombinasi input yang valid.")

with col2:
    st.subheader("Kurva Membership Pendapatan (Tahap 1)")
    _, var_pendapatan, _, _ = get_manual_fis()
    fig, ax = plt.subplots(figsize=(6,3))
    var_pendapatan.view(sim=None, ax=ax)
    st.pyplot(fig)