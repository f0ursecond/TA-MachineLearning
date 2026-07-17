# app/app.py
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="DSS Kelayakan Penerima Beasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich premium design
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-title {
        color: #1E3A8A;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #4B5563;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2563EB;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
        text-align: center;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_ml_components():
    model_path = 'models/best_model.pkl'
    agglo_path = 'models/agglo_model.pkl'
    prep_path = 'models/preprocessing.pkl'
    
    kmeans_model = None
    agglo_model = None
    preprocessing_data = None
    
    if os.path.exists(prep_path):
        with open(prep_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
            
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            kmeans_model = pickle.load(f)
            
    if os.path.exists(agglo_path):
        with open(agglo_path, 'rb') as f:
            agglo_model = pickle.load(f)
            
    return kmeans_model, agglo_model, preprocessing_data

kmeans, agglo, prep = load_ml_components()

# App Header
st.markdown("<h1 class='main-title'>🎓 Sistem Pendukung Keputusan Kelayakan Beasiswa</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Analisis Objektif Transparan Menggunakan Algoritma Unsupervised Clustering K-Means</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.magnific.com/512/11905/11905764.png", width=90)
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard EDA", "Model Demo (Form Input)", "Evaluasi Model", "Interpretasi & Insight", "Dokumentasi Metodologi"]
)

# Tampilkan warning jika model belum dilatih
if kmeans is None or prep is None:
    st.warning("⚠️ File model (`models/best_model.pkl` & `models/preprocessing.pkl`) belum ditemukan. Silakan jalankan `python src/train_model.py` terlebih dahulu di terminal untuk melatih model.")
elif agglo is None:
    st.info("ℹ️ File model proxy Agglomerative (`models/agglo_model.pkl`) belum ditemukan. Silakan jalankan kembali `python src/train_model.py` jika ingin membandingkan model.")

if menu == "Dashboard EDA":
    st.header("📊 Dashboard Analisis Eksploratif Data (EDA)")
    st.markdown("Berikut adalah visualisasi sebaran kriteria utama pendaftar beasiswa berdasarkan data historis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Heatmap Korelasi Kriteria")
        if os.path.exists("app/assets/heatmap_korelasi.png"):
            st.image("app/assets/heatmap_korelasi.png", width='stretch')
        else:
            st.info("Visualisasi `heatmap_korelasi.png` belum dibuat.")
            
    with col2:
        st.subheader("2. Boxplot Sebaran Kriteria Utama")
        if os.path.exists("app/assets/boxplot_perbandingan.png"):
            st.image("app/assets/boxplot_perbandingan.png", width='stretch')
        else:
            st.info("Visualisasi `boxplot_perbandingan.png` belum dibuat.")
            
    st.subheader("3. Analisis Hubungan IPK & Penghasilan Orang Tua")
    if os.path.exists("app/assets/scatter_ipk_penghasilan.png"):
        st.image("app/assets/scatter_ipk_penghasilan.png", width='stretch')
    else:
        st.info("Visualisasi `scatter_ipk_penghasilan.png` belum dibuat.")

elif menu == "Model Demo (Form Input)":
    st.header("⚙️ Demo Prediksi Kelayakan Pendaftar")
    st.markdown("Masukkan data profil pendaftar baru untuk menganalisis apakah pendaftar tergolong **LAYAK** atau **TIDAK LAYAK** menerima beasiswa berdasarkan model K-Means.")
    
    if kmeans is not None and prep is not None:
        le_dict = prep['le_dict']
        scaler = prep['scaler']
        fitur_kolom = prep['fitur_kolom']
        
        # Form Input
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Nama Lengkap Pendaftar", value="Budi Santoso")
                jk = st.selectbox("Jenis Kelamin", ["L", "P"])
                jarak = st.selectbox("Jarak Rumah ke Kampus", ["Dekat", "Jauh"])
                tahun_lulus = st.number_input("Tahun Lulus Sekolah Asal", min_value=2010, max_value=2026, value=2019)
                sks = st.number_input("SKS yang Diambil", min_value=1, max_value=24, value=20)
                organisasi = st.selectbox("Ikut Organisasi", ["Ikut", "Tidak"])
                
            with col2:
                ukm = st.selectbox("Ikut UKM", ["Ikut", "Tidak"])
                ipk = st.number_input("Indeks Prestasi Kumulatif (IPK)", min_value=0.0, max_value=4.0, value=3.30, step=0.01)
                
                # Pekerjaan Orang Tua (kita tampilkan list pekerjaan yang valid dari Encoder)
                job_classes = list(le_dict['Pekerjaan Orang Tua'].classes_)
                # Cari index default pekerjaan yang umum
                def_idx = job_classes.index('Petani') if 'Petani' in job_classes else 0
                pekerjaan = st.selectbox("Pekerjaan Orang Tua", job_classes, index=def_idx)
                
                penghasilan = st.selectbox("Penghasilan Orang Tua", ["Rendah", "Sedang", "Tinggi"])
                tanggungan = st.number_input("Jumlah Tanggungan Orang Tua (Anak)", min_value=1, max_value=12, value=3)
                
            submit_btn = st.form_submit_button("Analisis Kelayakan")
            
        if submit_btn:
            # Satukan data ke Dictionary
            raw_input = {
                'Jenis Kelamin': jk,
                'Jarak Tempat Tinggal kekampus (Km)': jarak,
                'Tahun Lulus': float(tahun_lulus),
                'SKS': float(sks),
                'Ikut Organisasi': organisasi,
                'Ikut UKM': ukm,
                'IPK': float(ipk),
                'Pekerjaan Orang Tua': pekerjaan,
                'Penghasilan': penghasilan,
                'Tanggungan': float(tanggungan)
            }
            
            # Encode
            encoded_input = {}
            for col, val in raw_input.items():
                if col in le_dict:
                    le = le_dict[col]
                    val_str = str(val)
                    if val_str in le.classes_:
                        encoded_input[col] = int(le.transform([val_str])[0])
                    else:
                        # Fallback jika ada kategori asing
                        encoded_input[col] = 0
                else:
                    encoded_input[col] = val
                    
            data_encoded = pd.DataFrame([encoded_input])[fitur_kolom]
            
            # Scale
            data_scaled = scaler.transform(data_encoded)
            
            # Predict
            pred_kmeans = kmeans.predict(data_scaled)[0]
            pred_agglo = agglo.predict(data_scaled)[0] if agglo is not None else None
            
            # Tampilkan Hasil
            st.subheader("🔍 Hasil Keputusan Model:")
            
            if pred_agglo is not None:
                col_km, col_ag = st.columns(2)
                
                with col_km:
                    st.markdown("### 🤖 K-Means Clustering (Optimal)")
                    if pred_kmeans == 0:
                        st.success(f"🎉 **{nama}** dinyatakan **LAYAK (Cluster 0)**")
                        st.markdown("""
                        * **Profil Klaster K-Means:**
                          - IPK Rata-rata: ~3.32
                          - Tanggungan: Banyak (rata-rata 3.16 anak)
                          - Ekonomi: Rendah-Sedang
                          - *Justifikasi:* Memenuhi prioritas sosio-ekonomi dan batas aman akademik.
                        """)
                    else:
                        st.warning(f"⚠️ **{nama}** dinyatakan **TIDAK LAYAK (Cluster 1)**")
                        st.markdown("""
                        * **Profil Klaster K-Means:**
                          - IPK Rata-rata: ~3.31
                          - Tanggungan: Sedikit (rata-rata 2.20 anak)
                          - Ekonomi: Sedang-Tinggi
                          - *Justifikasi:* Tidak memenuhi kriteria prioritas ekonomi lemah.
                        """)
                        
                with col_ag:
                    st.markdown("### 📈 Agglomerative Clustering")
                    if pred_agglo == 0:
                        st.success(f"🎉 **{nama}** dinyatakan **LAYAK (Cluster 0)**")
                        st.markdown("""
                        * **Profil Klaster Agglomerative:**
                          - IPK Rata-rata: ~3.34
                          - Tanggungan: Banyak (rata-rata 2.90 anak)
                          - Ekonomi: Menengah-Kebawah
                          - *Justifikasi:* Profil spasial hirarkis menunjukkan beban tanggungan relatif berat.
                        """)
                    else:
                        st.warning(f"⚠️ **{nama}** dinyatakan **TIDAK LAYAK (Cluster 1)**")
                        st.markdown("""
                        * **Profil Klaster Agglomerative:**
                          - IPK Rata-rata: ~3.29
                          - Tanggungan: Sedikit (rata-rata 2.23 anak)
                          - Ekonomi: Menengah-Keatas
                          - *Justifikasi:* Profil spasial hirarkis menunjukkan ekonomi lebih mandiri.
                        """)
            else:
                # Fallback jika model Agglomerative belum dilatih
                if pred_kmeans == 0:
                    st.success(f"🎉 Pendaftar **{nama}** dinyatakan **LAYAK (Cluster 0)** menerima beasiswa.")
                    st.markdown("""
                    * **Profil Klaster Kelayakan:**
                      - IPK Rata-rata: ~3.32
                      - Karakteristik Ekonomi: Tanggungan orang tua relatif banyak (rata-rata 3.16 anak), didominasi oleh penghasilan Rendah-Sedang.
                      - Justifikasi: Memenuhi kriteria sosio-ekonomi prioritas dan batas aman akademik.
                    """)
                else:
                    st.warning(f"⚠️ Pendaftar **{nama}** dinyatakan **TIDAK LAYAK (Cluster 1)** menerima beasiswa.")
                    st.markdown("""
                    * **Profil Klaster Kelayakan:**
                      - IPK Rata-rata: ~3.31
                      - Karakteristik Ekonomi: Tingkat ekonomi tergolong lebih mapan (penghasilan didominasi Sedang-Tinggi), tanggungan anak lebih sedikit (rata-rata 2.20 anak).
                      - Justifikasi: Tidak memenuhi kriteria prioritas kemiskinan yang dipersyaratkan oleh pemberi beasiswa.
                    """)
    else:
        st.info("Silakan latih model terlebih dahulu untuk dapat menggunakan fitur prediksi ini.")

elif menu == "Evaluasi Model":
    st.header("📈 Evaluasi Performa Model Clustering")
    st.markdown("Metrik evaluasi geometris dan confusion matrix model K-Means terhadap Ground Truth historis.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='metric-label'>Silhouette Score</div>
            <div class='metric-value'>0.1878</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <div class='metric-label'>Davies-Bouldin Index</div>
            <div class='metric-value'>2.1339</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card'>
            <div class='metric-label'>Optimal Cluster (k)</div>
            <div class='metric-value'>2</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Confusion Matrix (K-Means vs Ground Truth)")
    if os.path.exists("app/assets/confusion_matrix.png"):
        st.image("app/assets/confusion_matrix.png", width='stretch')
    else:
        st.info("Visualisasi `confusion_matrix.png` belum dibuat.")
        
    st.subheader("Metode Siku (Elbow Method)")
    if os.path.exists("app/assets/elbow_method.png"):
        st.image("app/assets/elbow_method.png", width='stretch')
    else:
        st.info("Visualisasi `elbow_method.png` belum dibuat.")

elif menu == "Interpretasi & Insight":
    st.header("💡 Analisis Anomali & Insight Bisnis")
    st.markdown("Berdasarkan perbandingan antara model klasterisasi objektif dan keputusan historis panitia seleksi, ditemukan kasus ketidaksesuaian keputusan (*mismatch rate*).")
    
    if os.path.exists("app/assets/visualisasi_anomali.png"):
        st.image("app/assets/visualisasi_anomali.png", width='stretch')
    else:
        st.info("Visualisasi `visualisasi_anomali.png` belum dibuat.")
        
    st.markdown("""
    ### ⚠️ Temuan Utama Analisis Anomali:
    Ditemukan sebanyak **286 data anomali (27.4%)** dari total 1.043 pendaftar beasiswa:
    
    1. 🔴 **Salah Terima (70 Pendaftar):**
       - **Karakteristik:** Rata-rata IPK sangat tinggi (**3.51**), namun tingkat ekonomi tergolong menengah ke atas (tanggungan sedikit, rata-rata **2.6** anak dan penghasilan didominasi sedang/tinggi).
       - **Insight:** Lolos karena aspek akademik (IPK) yang sangat menonjol sehingga mengabaikan aspek ekonomi sosial yang menjadi esensi utama beasiswa.
       
    2. 🟣 **Salah Tolak (216 Pendaftar):**
       - **Karakteristik:** Rata-rata IPK memenuhi syarat aman (**3.30**), namun kondisi ekonominya sangat membutuhkan bantuan (tanggungan rata-rata banyak, **3.1** anak, dan penghasilan didominasi rendah/sedang).
       - **Insight:** Merupakan sasaran target utama beasiswa yang terlewatkan (misclassified) oleh panitia seleksi, kemungkinan karena keterbatasan kuota atau bias subjektivitas seleksi manual.
    """)

elif menu == "Dokumentasi Metodologi":
    st.header("📖 Dokumentasi & Metodologi Proyek")
    st.markdown("""
    ### Alur Kerja Pembelajaran Mesin (ML Pipeline)
    Proyek ini mengadopsi standar metodologi data mining end-to-end:
    
    1. **Data Preprocessing & Cleaning:**
       - Penghapusan kolom bernilai kosong dan penyesuaian string.
       - Imputasi nilai kosong menggunakan median (numerik) dan modus (kategorikal).
       - Label encoding fitur kategorikal.
       - Standardisasi $z$-score untuk menyeimbangkan rentang skala fitur.
    2. **Eksperimen Klasterisasi:**
       - Membandingkan model **K-Means**, **Agglomerative Clustering**, dan **DBSCAN**.
       - Evaluasi jumlah klaster terbaik melalui *Elbow Method* (WCSS) dan *Silhouette Score*.
    3. **Evaluasi Anomali:**
       - Memetakan hasil klasterisasi biner ($k=2$) terhadap keputusan panitia aktual (*Ground Truth*).
       - Mengekstrak data anomali *Salah Terima* dan *Salah Tolak* untuk keperluan audit beasiswa.
    4. **Deployment:**
       - Mengekspor model terbaik menggunakan serialisasi pickle.
       - Membangun dashboard Streamlit ini sebagai antarmuka pendukung keputusan.
       
    ### Fitur-Fitur Analitis (10 Fitur):
    - **Akademik:** `IPK`, `SKS`
    - **Sosio-Ekonomi:** `Penghasilan`, `Tanggungan`, `Pekerjaan Orang Tua`
    - **Aktivitas:** `Ikut Organisasi`, `Ikut UKM`
    - **Demografis & Lainnya:** `Jenis Kelamin`, `Jarak Tempat Tinggal kekampus (Km)`, `Tahun Lulus`
    """)
