# LAPORAN PROYEK AKHIR: PEMBELAJARAN MESIN
## ANALISIS KESESUAIAN PENERIMA BEASISWA MENGGUNAKAN METODE CLUSTERING TERHADAP DATA GROUND TRUTH
---
* **Mata Kuliah:** Pembelajaran Mesin (Pembelajaran Tanpa Pengawasan / Unsupervised Learning)
* **Sifat:** Project (Capstone Project)
* **Kelompok / Kelas:** A11.4501 – A11.45XX
* **Dosen Pengampu:** Tim Dosen
* **Program Studi:** Teknik Informatika / Sistem Informasi / Ilmu Komputer
* **Universitas:** Universitas Dian Nuswantoro, Semarang

---

## 📌 DAFTAR ISI
1. [SOAL 1: PROBLEM DEFINITION & DATA ACQUISITION](#soal-1-problem-definition--data-acquisition)
   - 1.1 Latar Belakang dan Problem Statement
   - 1.2 Link atau Dokumentasi Sumber Dataset
   - 1.3 Statistik Deskriptif Awal
2. [SOAL 2: EXPLORATORY DATA ANALYSIS & PREPROCESSING](#soal-2-exploratory-data-analysis--preprocessing)
   - 2.1 Analisis Kualitas Data
   - 2.2 Feature Engineering & Preprocessing (Justifikasi Teknik)
   - 2.3 Pembagian Dataset (Train-Validation-Test Split)
   - 2.4 Visualisasi 5 Insights Paling Penting dari Data
3. [SOAL 3: MODELING & EVALUATION](#soal-3-modeling--evaluation)
   - 3.1 Implementasi Model Clustering
   - 3.2 Tuning Hyperparameter
   - 3.3 Tabel Perbandingan Performa Model
   - 3.4 Visualisasi Hasil Evaluasi
   - 3.5 Pemilihan Model Terbaik dan Justifikasi
   - 3.6 Penjelasan Detail Karakteristik Kelayakan & Analisis Anomali
4. [SOAL 4: DEPLOYMENT & STREAMLIT APPLICATION](#soal-4-deployment--streamlit-application)
   - 4.1 Deskripsi Aplikasi Streamlit & Fitur Utama
   - 4.2 Struktur Kode Aplikasi (`app.py`)
   - 4.3 Tautan Aplikasi & Screenshot Antarmuka (Placeholder)
5. [SOAL 5: DOCUMENTATION & PRESENTATION](#soal-5-documentation--presentation)
   - 5.1 Struktur Repositori Proyek
   - 5.2 Tautan Presentasi Video (YouTube)
   - 5.3 Kesimpulan & Rekomendasi
   - 5.4 Referensi

---

## SOAL 1: PROBLEM DEFINITION & DATA ACQUISITION
### 1.1 Latar Belakang dan Problem Statement
Dalam upaya pemerataan akses pendidikan, beasiswa merupakan salah satu instrumen penting bagi institusi pendidikan tinggi. Namun, proses seleksi penerima beasiswa secara manual atau historis sering kali dihadapkan pada tantangan subjektivitas, inkonsistensi keputusan, serta bias manusia dalam melakukan penilaian. Kriteria seleksi yang mencakup aspek akademik (seperti Indeks Prestasi Kumulatif atau IPK dan Satuan Kredit Semester atau SKS) serta aspek ekonomi (seperti penghasilan orang tua dan jumlah tanggungan) sering kali tidak ditimbang secara proporsional. Hal ini menyebabkan terjadinya ketidaksesuaian atau ketidakadilan seleksi, yang dapat dikategorikan menjadi dua jenis kesalahan utama: 'Salah Terima' (pendaftar yang secara finansial tergolong mampu namun berhasil lolos seleksi karena memiliki keunggulan akademik yang sangat menonjol) dan 'Salah Tolak' (pendaftar yang secara finansial sangat membutuhkan bantuan dan memenuhi syarat minimal akademik, namun tersingkir dari proses seleksi karena pembobotan yang kurang seimbang). 

Untuk mengatasi permasalahan tersebut, pendekatan data-driven dengan memanfaatkan pembelajaran mesin tanpa pengawasan (unsupervised learning) seperti clustering menjadi solusi objektif yang sangat relevan. Melalui teknik pengelompokan (clustering) pendaftar berdasarkan profil akademik dan kondisi ekonomi mereka secara simultan, kita dapat mengidentifikasi pola alami kelayakan pendaftar tanpa adanya intervensi subjektif dari evaluator manusia. Hasil pengelompokan objektif ini kemudian dibandingkan dengan keputusan aktual seleksi (Ground Truth) untuk mengekstrak anomali-anomali keputusan yang terjadi. Dengan mendeteksi dan menganalisis karakteristik pendaftar pada kelompok anomali tersebut, pihak manajemen universitas dapat mengevaluasi kebijakan seleksi, memperbaiki formula pembobotan kriteria, dan membangun Sistem Pendukung Keputusan (DSS) yang lebih transparan, objektif, dan tepat sasaran.

### 1.2 Link atau Dokumentasi Sumber Dataset
* **Nama File Dataset:** `dataset.csv`
* **Lokasi Penyimpanan:** Root direktori project (`./dataset.csv`).
* **Ukuran Dataset Mentah:** 1.043 data pendaftar beasiswa dengan 26 kolom (termasuk 11 kolom kosong `Unnamed`).
* **Tautan Sumber Data:** *(Silakan ganti/tambahkan link repositori GitHub Anda jika sudah dipublikasikan, contoh: `https://github.com/username/capstone-project-data-mining/blob/main/dataset.csv`)*

### 1.3 Statistik Deskriptif Awal
Berikut adalah ringkasan statistik deskriptif awal dari dataset:
* **Ukuran Data:** 1.043 baris (setelah pembersihan baris kosong di baris terbawah) dan 26 kolom awal.
* **Jumlah Fitur Terpilih:** 10 Fitur yang digunakan untuk clustering (setelah menghapus identitas non-analitik dan kolom kosong).
* **Tipe Data Fitur:**
  * **Variabel Kategorikal:**
    * `Jenis Kelamin` (Object - L/P)
    * `Jarak Tempat Tinggal kekampus (Km)` (Object - Dekat/Jauh)
    * `Ikut Organisasi` (Object - Ikut/Tidak)
    * `Ikut UKM` (Object - Ikut/Tidak)
    * `Pekerjaan Orang Tua` (Object - 120+ variasi teks pekerjaan)
    * `Penghasilan` (Object - Rendah, Sedang, Tinggi)
  * **Variabel Numerik:**
    * `Tahun Lulus` (Float64 - Tahun kelulusan sekolah asal)
    * `SKS` (Float64 - Jumlah SKS yang diambil)
    * `IPK` (Float64 - Indeks Prestasi Kumulatif)
    * `Tanggungan` (Float64 - Jumlah tanggungan anak orang tua)
* **Status Beasiswa (Target aktual / Ground Truth):** 272 pendaftar berstatus "Terima" (26.08%) dan 771 pendaftar berstatus "Tidak" (73.92%).

---

## SOAL 2: EXPLORATORY DATA ANALYSIS & PREPROCESSING
### 2.1 Analisis Kualitas Data
1. **Missing Values (Nilai Kosong):**
   * Deteksi awal menunjukkan kolom `Unnamed: 15` sampai `Unnamed: 25` bernilai `NaN` (kosong) 100%. Kolom-kolom ini langsung dihapus.
   * Terdapat 1 baris di bagian paling bawah yang bernilai kosong di hampir semua kolom, baris ini berhasil dibuang.
   * Untuk sisa missing values pada baris valid, dilakukan **imputasi**: data numerik diisi dengan nilai *median* dari kolom tersebut, sedangkan data kategorikal diisi dengan *mode* (modus/nilai tersering).
2. **Outliers (Pencilan):**
   * Deteksi sebaran boxplot pada variabel `IPK` menunjukkan sebaran data yang cukup normal dengan beberapa pencilan di bagian bawah (IPK sangat rendah), namun tetap dipertahaman karena merepresentasikan kondisi pendaftar nyata.
3. **Duplikat:**
   * Ditemukan 1 baris duplikat sebelum pembersihan yang kemudian berhasil dihapus, menyisakan 1.043 data unik.
4. **Inkonsistensi String:**
   * Dilakukan *stripping* whitespace pada seluruh nama kolom dan nilai teks (`df_clean.columns = df_clean.columns.str.strip()`) guna menghindari inkonsistensi pencocokan string saat analisis maupun encoding.

### 2.2 Feature Engineering & Preprocessing (Justifikasi Teknik)
Langkah preprocessing yang diimplementasikan pada dataset meliputi:
1. **Seleksi Fitur (Feature Selection):**
   * Menghapus kolom identitas non-analitik: `No`, `Nama Lengkap`, `Prodi`, `Asal Sekolah`, serta kolom target `Status Beasiswa` dan `Ground_Truth`.
   * Tersisa 10 fitur analitis: `['Jenis Kelamin', 'Jarak Tempat Tinggal kekampus (Km)', 'Tahun Lulus', 'SKS', 'Ikut Organisasi', 'Ikut UKM', 'IPK', 'Pekerjaan Orang Tua', 'Penghasilan', 'Tanggungan']`.
2. **Encoding Data Kategorikal:**
   * Menggunakan `LabelEncoder` dari `scikit-learn` untuk mengonversi data teks kategorikal menjadi numerik diskret.
   * *Justifikasi:* Algoritma klasterisasi berbasis jarak membutuhkan representasi numerik untuk menghitung kedekatan antar data.
3. **Standardisasi Skala Fitur (Feature Scaling):**
   * Menggunakan `StandardScaler` untuk melakukan normalisasi *z-score* (rata-rata = 0, standar deviasi = 1).
   * *Justifikasi:* Algoritma clustering (seperti K-Means dan Agglomerative) sangat sensitif terhadap skala data karena menggunakan perhitungan jarak Euclidean. Jika tidak distandardisasi, fitur dengan nilai nominal besar (misalnya `Tahun Lulus` atau `Pekerjaan Orang Tua` ter-encode) akan mendominasi perhitungan jarak dibandingkan fitur berkisar kecil seperti `IPK` (0.00 - 4.00).

### 2.3 Pembagian Dataset (Train-Validation-Test Split)
* **Justifikasi Analisis Clustering:** 
  Dalam pemodelan unsupervised clustering, data tidak dibagi ke dalam *train*, *validation*, dan *test split* secara tradisional seperti pada pemodelan supervised learning (klasifikasi/regresi). Hal ini karena clustering mencari pola alami (*inherent patterns*) dari seluruh populasi data yang ada untuk membentuk kelompok-kelompok baru. Seluruh 1.043 data yang telah distandardisasi langsung dimasukkan ke dalam algoritma clustering untuk melatih dan mengevaluasi kualitas cluster.
* **Justifikasi Tambahan untuk Implementasi Klasifikasi Supervised di Masa Depan:**
  Jika di kemudian hari model klasterisasi ini diubah menjadi klasifikasi terbimbing (supervised learning) dengan label cluster sebagai targetnya, maka dataset disarankan untuk dibagi dengan rasio **80% training set**, **10% validation set**, dan **10% test set** (atau **70% train** dan **30% test**) guna mengevaluasi tingkat generalisasi model klasifikasi terhadap data baru.

### 2.4 Visualisasi 5 Insights Paling Penting dari Data
Berikut adalah visualisasi insights kunci yang diekstrak dari data:

#### Insight 1: Heatmap Korelasi Antar Fitur (`heatmap_korelasi.png`)
![Heatmap Korelasi](/app/assets/heatmap_korelasi.png)
* *Keterangan:* Peta korelasi antar fitur numerik dan kategorikal ter-encode.
* *Insight:* Memberikan gambaran hubungan linier antar variabel. Fitur ekonomi seperti Penghasilan dan Tanggungan memiliki hubungan korelasi moderat terhadap penentuan beasiswa aktual, sementara IPK memiliki korelasi yang cenderung rendah dalam matriks korelasi linier global, yang mengindikasikan hubungan yang bersifat non-linier atau adanya seleksi multikriteria yang kompleks.

#### Insight 2: Sebaran Fitur Berdasarkan Status Beasiswa (`boxplot_perbandingan.png`)
![Boxplot Perbandingan](/app/assets/boxplot_perbandingan.png)
* *Keterangan:* Distribusi IPK, Tanggungan, SKS, dan Penghasilan untuk masing-masing kelompok penerimaan aktual.
* *Insight:* Terlihat bahwa pendaftar yang diterima beasiswa secara historis memiliki sebaran IPK yang sedikit lebih tinggi, namun sebaran variabel ekonomi (tanggungan dan penghasilan) antara kelompok yang diterima dan ditolak masih sangat tumpang tindih (overlap). Hal ini mengindikasikan adanya inkonsistensi keputusan seleksi manual.

#### Insight 3: Titik Elbow dan Silhouette Score (`elbow_method.png`)
![Elbow Method](/app/assets/elbow_method.png)
* *Keterangan:* Grafik WCSS (Within-Cluster Sum of Squares) dan Silhouette Score untuk rentang $k \in [2, 10]$.
* *Insight:* Penurunan WCSS paling melambat setelah $k=2$. Evaluasi Silhouette Score juga menunjukkan puncak tertinggi pada $k=2$ (0.1878), membuktikan bahwa struktur data pendaftar secara alami paling optimal jika dibagi menjadi dua kelompok (Layak vs Tidak Layak).

#### Insight 4: Korelasi Scatter IPK vs Penghasilan (`scatter_ipk_penghasilan.png`)
![Scatter IPK vs Penghasilan](/app/assets/scatter_ipk_penghasilan.png)
* *Keterangan:* Visualisasi koordinat pendaftar berdasarkan IPK (sumbu $x$) dan Penghasilan (sumbu $y$), dikelompokkan berdasarkan kategori kesesuaian seleksi.
* *Insight:* Plot ini memperlihatkan klaster pendaftar yang mengalami anomali. Kelompok "Salah Terima" dominan berada di kuadran kanan atas (IPK tinggi, penghasilan tinggi), sedangkan kelompok "Salah Tolak" dominan berada di kuadran kiri bawah (IPK cukup, penghasilan rendah/sedang).

#### Insight 5: Sebaran Distribusi Data Anomali (`visualisasi_anomali.png`)
![Visualisasi Anomali](/app/assets/visualisasi_anomali.png)
* *Keterangan:* Grafik distribusi kategori kesesuaian keputusan (Benar: Tolak, Benar: Terima, Salah Tolak, Salah Terima).
* *Insight:* Total data yang terindikasi anomali keputusan adalah 286 pendaftar (27.4%). Anomali didominasi oleh kasus "Salah Tolak" sebanyak 216 orang, jauh lebih banyak dibanding kasus "Salah Terima" (70 orang). Ini menunjukkan bahwa sistem seleksi manual historis cenderung terlalu ketat atau salah sasaran dalam menolak pendaftar yang secara ekonomi sangat layak.

---

## SOAL 3: MODELING & EVALUATION
### 3.1 Implementasi Model Clustering
Proyek ini menguji tiga algoritma klasterisasi yang berbeda untuk menemukan pengelompokan kelayakan terbaik:
1. **K-Means Clustering:** Mengelompokkan data berdasarkan jarak Euclidean terkecil terhadap pusat klaster (centroid).
2. **Agglomerative Hierarchical Clustering:** Pendekatan *bottom-up* yang menggabungkan pasangan klaster terdekat secara hierarkis menggunakan linkage 'Ward'.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Mengelompokkan data berdasarkan kepadatan density data point dalam radius `eps`.

### 3.2 Tuning Hyperparameter
* **K-Means & Agglomerative:** Menyetel jumlah klaster $k=2$ berdasarkan hasil analisis Elbow Method dan Silhouette Score optimal.
* **DBSCAN:** Dilakukan tuning parameter radius ketetanggaan `eps` dan jumlah sampel minimum `min_samples`. Kombinasi terbaik diperoleh pada `eps=2.5` dan `min_samples=5`.

### 3.3 Tabel Perbandingan Performa Model
Berikut adalah perbandingan performa ketiga model yang diuji:

| Metode Clustering | Silhouette Score | Davies-Bouldin Index | Keterangan / Hasil |
| :--- | :---: | :---: | :--- |
| **K-Means ($k=2$)** | **0.1878** | **2.1339** | **Model Terbaik (Dipilih)**. Berhasil membagi data secara seimbang (Cluster 0: 418, Cluster 1: 625). |
| **Agglomerative ($k=2$)** | 0.1381 | 2.3658 | Performa di bawah K-Means. Pembagian data kurang merata (Cluster 0: 558, Cluster 1: 485). |
| **DBSCAN (eps=2.5, min=5)** | 0.0000 | - | **Gagal**. Menghasilkan 1 klaster dominan (1.032 data) dan 11 noise. Tidak mampu memisahkan kelompok kelayakan. |

### 3.4 Visualisasi Hasil Evaluasi
Untuk mengevaluasi kualitas cluster terhadap Ground Truth (keputusan aktual beasiswa), kami menggunakan **Confusion Matrix** (`confusion_matrix.png`):

![Confusion Matrix](/app/assets/confusion_matrix.png)
* *Insight dari Confusion Matrix:* K-Means ($k=2$) menunjukkan kecocokan yang kuat dengan keputusan asli pada kelompok penolakan beasiswa (555 data terklasifikasi benar sebagai tidak layak/ditolak). Namun, terdapat sebaran anomali yang jelas pada kuadran cross-tabulation.

### 3.5 Pemilihan Model Terbaik dan Justifikasi
**K-Means ($k=2$) dipilih sebagai model terbaik** dengan justifikasi sebagai berikut:
1. **Silhouette Score Tertinggi (0.1878):** Menunjukkan kerapatan anggota di dalam klaster dan kerenggangan antar klaster yang paling optimal dibandingkan Agglomerative (0.1381).
2. **Davies-Bouldin Index Terendah (2.1339):** Menandakan pemisahan klaster yang lebih baik.
3. **Pembagian Klaster yang Logis:** Pembagian $k=2$ menghasilkan representasi biner yang secara praktis sangat cocok untuk menguji sistem seleksi beasiswa (Layak vs Tidak Layak). DBSCAN gagal karena keterbatasan data yang memiliki variansi tinggi sehingga mengelompokkan hampir seluruh data ke dalam satu kelompok besar.

### 3.6 Penjelasan Detail Karakteristik Kelayakan & Analisis Anomali
Berdasarkan rata-rata nilai fitur dari kelompok K-Means yang terbentuk, didapatkan profil objektif sebagai berikut:
* **Cluster 0 (LAYAK):** Karakteristik ekonomi lemah (persentase penghasilan rendah 33.25%), jumlah tanggungan orang tua banyak (rata-rata 3.16 anak), dengan rata-rata IPK 3.32.
* **Cluster 1 (TIDAK LAYAK):** Karakteristik ekonomi lebih mampu (persentase penghasilan rendah hanya 20.48%), tanggungan orang tua sedikit (rata-rata 2.20 anak), dengan rata-rata IPK 3.31.

#### Evaluasi Komparasi (Crosstab): Ground Truth vs K-Means
| Kategori Analisis | K-Means: Cluster 0 (Layak) | K-Means: Cluster 1 (Tidak Layak) | Total Aktual |
|---|---|---|---|
| **GT: Terima Beasiswa** | **202** (Benar: Terima) | **70** (Salah Terima / Anomali) | **272** |
| **GT: Tidak Terima** | **216** (Salah Tolak / Anomali) | **555** (Benar: Tolak) | **771** |
| **Total Klaster** | **418** | **625** | **1.043** |

#### Analisis Data Anomali (Total: 286 pendaftar / 27.4%)
1. **Anomali 'Salah Terima' (70 Pendaftar):**
   * *Kondisi:* Secara aktual Diterima, namun secara clustering masuk kategori Tidak Layak.
   * *Karakteristik:* Rata-rata IPK sangat tinggi (**3.51**), namun secara ekonomi tergolong mampu (didominasi penghasilan Sedang dan Tinggi) dengan tanggungan keluarga yang kecil (rata-rata **2.6** anak).
   * *Justifikasi:* Kelompok ini diuntungkan oleh keunggulan akademik mereka yang menonjol sehingga melompati kriteria ketidakmampuan ekonomi pada seleksi manual.
2. **Anomali 'Salah Tolak' (216 Pendaftar):**
   * *Kondisi:* Secara aktual Ditolak, namun secara clustering masuk kategori Layak.
   * *Karakteristik:* Rata-rata IPK memenuhi standar aman (**3.30**), namun kondisi ekonomi sangat mendesak (didominasi penghasilan Rendah dan Sedang) dengan tanggungan keluarga yang berat (rata-rata **3.1** anak).
   * *Justifikasi:* Kelompok ini adalah kelompok sasaran utama beasiswa sosial yang terlewatkan (misclassified) karena kuota seleksi manual yang terbatas atau sistem pembobotan yang kurang sensitif terhadap beban ekonomi pendaftar.

---

## SOAL 4: DEPLOYMENT & STREAMLIT APPLICATION
### 4.1 Deskripsi Aplikasi Streamlit & Fitur Utama
Aplikasi web interaktif dibangun menggunakan framework **Streamlit** untuk mendemonstrasikan sistem pendukung keputusan seleksi beasiswa secara real-time. Aplikasi ini mencakup 5 fitur utama sesuai dengan ketentuan UAS:
1. **Dashboard EDA:** Visualisasi interaktif grafik distribusi fitur pendaftar (IPK, Penghasilan, Tanggungan) dan matriks korelasi.
2. **Model Demo (Form Input):** Antarmuka input interaktif bagi pengguna (nama pendaftar, IPK, penghasilan, tanggungan, dll.) untuk mendapatkan hasil prediksi kelayakan langsung ("LAYAK" atau "TIDAK LAYAK") dari model K-Means terbaik yang sudah disimpan.
3. **Evaluasi Model:** Menampilkan metrik performa model (Silhouette Score, Davies-Bouldin) beserta visualisasi Confusion Matrix perbandingan terhadap Ground Truth historis.
4. **Interpretasi Hasil & Insight Bisnis:** Penjelasan detail profil klaster kelayakan serta grafik analisis anomali.
5. **Dokumentasi Aplikasi:** Petunjuk langkah-langkah penggunaan aplikasi serta ringkasan metodologi.

### 4.2 Struktur Kode Aplikasi (`app.py`)
Berikut adalah rancangan/template kode utama `app.py` untuk deployment Streamlit:

```python
# app.py - RANCANGAN APLIKASI WEB STREAMLIT
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DSS Beasiswa", layout="wide")

# Load model dan preprocessing pipeline jika ada
# with open('models/best_model.pkl', 'rb') as f:
#     model = pickle.load(f)

st.title("🎓 Sistem Analisis Kelayakan & Pendukung Keputusan Penerima Beasiswa")
st.markdown("Aplikasi berbasis Machine Learning (Clustering K-Means) untuk mengoptimalkan ketepatan sasaran penerima beasiswa.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigasi Menu", 
                            ["Dashboard EDA", "Prediksi Kelayakan (Demo)", "Evaluasi Model", "Interpretasi & Insight", "Dokumentasi"])

if menu == "Dashboard EDA":
    st.header("📊 Dashboard Analisis Data Eksploratif (EDA)")
    # Tampilkan grafik sebaran data, boxplot, dan heatmap korelasi
    st.image("heatmap_korelasi.png", caption="Heatmap Korelasi Fitur")
    st.image("boxplot_perbandingan.png", caption="Boxplot Sebaran Kriteria Utama")

elif menu == "Prediksi Kelayakan (Demo)":
    st.header("⚙️ Form Prediksi Kelayakan Pendaftar")
    col1, col2 = st.columns(2)
    with col1:
        nama = st.text_input("Nama Lengkap Pendaftar")
        ipk = st.number_input("Indeks Prestasi Kumulatif (IPK)", min_value=0.0, max_value=4.0, value=3.30, step=0.01)
        tanggungan = st.number_input("Jumlah Tanggungan Orang Tua", min_value=0, max_value=10, value=3)
    with col2:
        penghasilan = st.selectbox("Tingkat Penghasilan Orang Tua", ["Rendah", "Sedang", "Tinggi"])
        organisasi = st.selectbox("Keaktifan Organisasi", ["Ikut", "Tidak"])
        ukm = st.selectbox("Keaktifan UKM", ["Ikut", "Tidak"])
        
    if st.button("Analisis Kelayakan"):
        # Logika prediksi berdasarkan model K-Means
        # dummy logic for format
        if ipk >= 3.0 and (penghasilan == "Rendah" or tanggungan >= 3):
            st.success(f"Hasil Analisis: Pendaftar {nama} dinyatakan **LAYAK (Cluster 0)** menerima beasiswa.")
        else:
            st.warning(f"Hasil Analisis: Pendaftar {nama} dinyatakan **TIDAK LAYAK (Cluster 1)** menerima beasiswa.")

elif menu == "Evaluasi Model":
    st.header("📈 Evaluasi Performa Model")
    st.metric(label="K-Means Silhouette Score", value="0.1878")
    st.metric(label="K-Means Davies-Bouldin Index", value="2.1339")
    st.image("confusion_matrix.png", caption="Confusion Matrix Terhadap Ground Truth")

elif menu == "Interpretasi & Insight":
    st.header("💡 Analisis Anomali & Insight Bisnis")
    st.image("visualisasi_anomali.png", caption="Distribusi Kasus Anomali Keputusan Seleksi")

elif menu == "Dokumentasi":
    st.header("📖 Dokumentasi & Cara Penggunaan")
    st.markdown("""
    **Metodologi:**
    1. Preprocessing data menggunakan `LabelEncoder` dan `StandardScaler`.
    2. Pemodelan unsupervised menggunakan K-Means dengan $k=2$.
    3. Evaluasi anomali keputusan manual dengan membandingkan prediksi klaster kelayakan terhadap data historis (Ground Truth).
    """)
```

### 4.3 Tautan Aplikasi & Screenshot Antarmuka (Placeholder)
* **Tautan Aplikasi Live (Streamlit Cloud):** `https://share.streamlit.io/username/repo-name/main/app.py` *(Placeholder)*
* **Screenshot Antarmuka Aplikasi:** *(Silakan masukkan file gambar screenshot aplikasi web Streamlit Anda ke dalam repositori dengan nama `screenshot_app.png` lalu tautkan di bawah ini)*
  
  ![Screenshot Aplikasi](screenshot_app.png) *(Placeholder)*

---

## SOAL 5: DOCUMENTATION & PRESENTATION
### 5.1 Struktur Repositori Proyek
Repositori proyek ini dirancang rapi dengan struktur folder standar sebagai berikut:
```
capstone-project-data-mining/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Dataset mentah asli
│   └── processed/
│       └── data_anomali.csv            # Data pendaftar yang terdeteksi anomali
│
├── notebooks/
│   └── 01_eda.ipynb                    # Notebook analisis data (analisis_beasiswa.ipynb)
│
├── src/
│   ├── data_preprocessing.py           # Ekstraksi fungsi pembersihan & scaling
│   ├── train_model.py                  # Ekstraksi fungsi pelatihan model
│   └── utils.py                        # Fungsi utilitas pembantu
│
├── models/
│   ├── best_model.pkl                  # Model K-Means terekspor (Pickle)
│   └── preprocessing.pkl               # Pipeline standardizer dan label encoder
│
├── app/
│   ├── app.py                          # Kode aplikasi web Streamlit utama
│   └── assets/
│       ├── boxplot_perbandingan.png    # File gambar visualisasi
│       ├── confusion_matrix.png
│       ├── elbow_method.png
│       ├── heatmap_korelasi.png
│       ├── scatter_ipk_penghasilan.png
│       └── visualisasi_anomali.png
│
├── reports/
│   └── final_report.pdf                # Ekspor file laporan akhir (PDF)
│
├── requirements.txt                    # Daftar dependensi library Python
├── README.md                           # Dokumentasi ringkas proyek
├── LAPORAN.md                          # Laporan teknis lengkap (Dokumen ini)
└── .gitignore                          # File exclude untuk git
```

### 5.2 Tautan Presentasi Video (YouTube)
* **Tautan Video Youtube:** `https://youtu.be/example_video_id` *(Placeholder - Silakan unggah rekaman presentasi proyek akhir Anda berdurasi 5-10 menit ke YouTube dan tautkan kodenya di sini)*

### 5.3 Kesimpulan & Rekomendasi
#### Kesimpulan Utama
1. **Objektivitas Data-Driven:** Metode clustering K-Means ($k=2$) terbukti berhasil membagi kelompok pendaftar secara objektif menjadi dua klaster kelayakan (Layak vs Tidak Layak) berdasarkan data sosio-ekonomi dan akademik secara bersamaan.
2. **Indikasi Mismatch Tinggi:** Ditemukan tingkat ketidaksesuaian keputusan seleksi sebesar **27.4% (286 pendaftar)**. 
3. **Analisis Pola Kesalahan:** 
   * **Salah Terima (70 pendaftar):** Lolos karena nilai IPK yang sangat tinggi (rata-rata 3.51) meskipun secara profil ekonomi tidak mendesak.
   * **Salah Tolak (216 pendaftar):** Ditolak kemungkinan besar karena batas kuota/subjektivitas, padahal memiliki beban ekonomi terberat (rata-rata 3.1 anak) dengan IPK yang masih di atas batas aman (3.30).

#### Rekomendasi Sistemik
1. **Penyeimbangan Kriteria Seleksi:** Institusi perlu mengevaluasi bobot seleksi manual agar kriteria ekonomi (seperti penghasilan dan tanggungan) serta kriteria akademik (IPK) dinilai secara seimbang (proporsional), bukan saling meniadakan.
2. **Penerapan Sistem Pendukung Keputusan (DSS):** Integrasikan hasil pemodelan K-Means ini sebagai penyaring tahap awal (*automated pre-screening*) untuk mendeteksi pendaftar yang sangat layak secara data sebelum dilakukan review berkas manual.
3. **Audit Keputusan Berkala:** Menggunakan daftar anomali (`data_anomali.csv`) sebagai subjek verifikasi lapangan (*field audit*) guna meminimalisir salah sasaran alokasi dana bantuan pendidikan di masa depan.

### 5.4 Referensi
1. Han, J., Kamber, M., & Jian, P. (2012). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
4. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.
