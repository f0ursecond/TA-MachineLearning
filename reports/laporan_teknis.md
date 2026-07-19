# LAPORAN TEKNIS: PEMBELAJARAN MESIN
## ANALISIS KESESUAIAN PENERIMA BEASISWA MENGGUNAKAN METODE CLUSTERING TERHADAP DATA GROUND TRUTH
---
* **Mata Kuliah:** Pembelajaran Mesin (Unsupervised Learning)
* **Sifat:** Project (Capstone Project)
* **Program Studi:** Teknik Informatika
* **Tahun Akademik:** 2026

---

# 1. PENDAHULUAN DAN LATAR BELAKAR
Dalam upaya pemerataan akses pendidikan, program beasiswa merupakan salah satu instrumen penting bagi perguruan tinggi untuk mendukung mahasiswa yang layak secara akademik maupun ekonomi. Namun, proses seleksi manual historis yang dilakukan oleh panitia seleksi sering kali menghadapi tantangan subjektivitas, inkonsistensi keputusan, serta bias manusia dalam melakukan penilaian berkas. Kriteria seleksi yang mencakup aspek akademik (seperti IPK dan SKS) serta aspek ekonomi (seperti penghasilan orang tua dan jumlah tanggungan) sering kali tidak ditimbang secara proporsional dan transparan. 

Hal ini menyebabkan terjadinya ketidaksesuaian keputusan seleksi di lapangan, yang dapat dikategorikan menjadi dua jenis kesalahan utama:
* **Salah Terima:** Pendaftar yang secara profil sosio-ekonomi tergolong mampu, namun dinyatakan diterima beasiswa karena faktor keunggulan akademik (IPK) yang sangat menonjol.
* **Salah Tolak:** Pendaftar yang secara profil sosio-ekonomi sangat layak dibantu (penghasilan rendah & tanggungan banyak) dan memenuhi syarat batas aman akademik, namun dinyatakan ditolak akibat keterbatasan kuota atau bias subjektif panitia.

Untuk mengatasi permasalahan tersebut, pendekatan data-driven dengan memanfaatkan pembelajaran mesin tanpa pengawasan (*unsupervised machine learning*) seperti clustering menjadi solusi objektif yang sangat relevan. Dengan mengabaikan label keputusan historis dan membiarkan algoritma mengelompokkan data pendaftar murni berdasarkan kemiripan profil kriteria mereka, kita dapat mengidentifikasi pola alami kelayakan pendaftar. Hasil pengelompokan objektif ini kemudian dibandingkan dengan keputusan aktual (*Ground Truth*) untuk mengekstrak anomali-anomali keputusan yang terjadi. Dengan mendeteksi dan menganalisis karakteristik pendaftar pada kelompok anomali tersebut, pihak manajemen universitas dapat mengevaluasi kebijakan seleksi dan membangun Sistem Pendukung Keputusan (DSS) yang lebih objektif dan tepat sasaran.

---

# 2. METODOLOGI (ALUR KERJA DAN TEKNIK YANG DIGUNAKAN)
Penelitian ini menggunakan dataset riil berisi 1.043 pendaftar beasiswa dengan 10 kriteria analitik utama: `Jenis Kelamin`, `Jarak Tempat Tinggal kekampus (Km)`, `Tahun Lulus`, `SKS`, `Ikut Organisasi`, `Ikut UKM`, `IPK`, `Pekerjaan Orang Tua`, `Penghasilan`, dan `Tanggungan`.

### 2.1 Alur Kerja Proyek (Workflow)
Proses analisis dan pemodelan dilakukan secara terstruktur melalui tahapan berikut:
1. **Pembersihan Data:** Menghapus kolom kosong, membuang data duplikat, dan melakukan imputasi *missing values* menggunakan median untuk fitur numerik serta modus untuk fitur kategorikal.
2. **Feature Engineering:** Mengonversi data kategorikal teks menjadi numerik diskret menggunakan `LabelEncoder` dan melakukan standardisasi skala fitur menggunakan `StandardScaler` (z-score normalization) agar fitur dengan rentang nominal besar tidak mendominasi perhitungan jarak.
3. **Penentuan Cluster Optimal:** Mengevaluasi jumlah klaster ($k$) terbaik menggunakan *Elbow Method* ( Within-Cluster Sum of Squares) dan Silhouette Score pada rentang $k \in [2, 10]$.
4. **Pemodelan Clustering:** Menguji tiga algoritma klasterisasi (K-Means, Agglomerative Clustering, dan DBSCAN) untuk mengelompokkan kelayakan pendaftar.
5. **Evaluasi dan Komparasi:** Memilih model terbaik berdasarkan metrik evaluasi internal, memetakan karakteristik klaster yang terbentuk (Layak vs Tidak Layak), dan membandingkannya terhadap keputusan aktual seleksi (*Ground Truth*) menggunakan *crosstab analysis* untuk mendeteksi anomali.

### 2.2 Teknik dan Algoritma yang Digunakan
* **K-Means Clustering:** Algoritma berbasis partisi yang meminimalkan variansi dalam klaster dengan mengelompokkan data ke centroid terdekat berdasarkan jarak Euclidean.
* **Agglomerative Hierarchical Clustering:** Pendekatan klasterisasi hierarkis bottom-up yang secara rekursif menggabungkan klaster terdekat berdasarkan kriteria linkage Ward untuk meminimalkan variansi total.
* **DBSCAN (Density-Based Spatial Clustering):** Algoritma berbasis kepadatan spasial yang memisahkan area padat dari area renggang (noise) menggunakan parameter epsilon (`eps`) dan minimum points (`min_samples`).

---

# 3. HASIL DAN ANALISIS
Bagian ini memaparkan hasil eksperimen pemodelan clustering, karakteristik klaster, serta temuan kasus anomali seleksi.

### 3.1 Penentuan Jumlah Klaster Optimal
Berdasarkan grafik Elbow Method dan evaluasi Silhouette Score, pembagian data pendaftar beasiswa secara alami paling optimal jika dibagi menjadi **2 klaster (k=2)**, yang merepresentasikan kelompok "Layak" dan "Tidak Layak".

![Elbow Method](/app/assets/elbow_method.png)
*Gambar 1: Evaluasi Elbow Method dan Silhouette Score untuk rentang klaster 2-10*

### 3.2 Perbandingan Performa Algoritma
Ketiga model diuji pada data yang telah distandardisasi dengan hasil sebagai berikut:

| Metode Clustering | Silhouette Score | Davies-Bouldin Index | Keterangan |
| :--- | :---: | :---: | :--- |
| **K-Means (k=2)** | **0.1878** | **2.1339** | **Model Terbaik (Dipilih)** |
| **Agglomerative (k=2)** | 0.1381 | 2.3658 | Performa di bawah K-Means |
| **DBSCAN (eps=2.5, min=5)** | 0.0000 | - | Terlalu sensitif (1 cluster dominan & 11 noise) |

Model **K-Means dengan k=2** terpilih sebagai model terbaik karena menghasilkan Silhouette Score tertinggi (0.1878) dan Davies-Bouldin Index terendah (2.1339), yang menunjukkan pembagian klaster yang paling tegas dan konsisten.

### 3.3 Karakteristik Klaster Kelayakan
Berdasarkan nilai rata-rata fitur pada kedua klaster yang terbentuk:
* **Cluster 0 (LAYAK):** Memiliki rata-rata IPK 3.32, persentase ekonomi rendah yang signifikan (33.25%), serta beban keluarga tinggi dengan rata-rata tanggungan orang tua sebanyak **3.16 anak**.
* **Cluster 1 (TIDAK LAYAK):** Memiliki rata-rata IPK 3.31, profil ekonomi yang lebih mapan (hanya 20.48% ekonomi rendah), serta rata-rata tanggungan yang sedikit yaitu **2.20 anak**.

### 3.4 Komparasi Terhadap Ground Truth dan Analisis Anomali
Hasil clustering K-Means dibandingkan dengan keputusan seleksi aktual (*Ground Truth*) menggunakan tabel silang (*crosstab*):

| Kategori Analisis | K-Means: Cluster 0 (Layak) | K-Means: Cluster 1 (Tidak Layak) | Total |
| :--- | :---: | :---: | :---: |
| **Ground Truth: Terima Beasiswa** | **202** (Benar: Terima) | **70** (Salah Terima / Anomali) | **272** |
| **Ground Truth: Tidak Terima** | **216** (Salah Tolak / Anomali) | **555** (Benar: Tolak) | **771** |
| **Total** | **418** | **625** | **1.043** |

Ditemukan sebanyak **286 kasus anomali (27.4%)** dari total 1.043 pendaftar. Anomali ini dikategorikan menjadi dua jenis:
1. **Kasus Salah Terima (70 Pendaftar):** Kelompok ini secara profil sosio-ekonomi mampu (tanggungan sedikit, penghasilan orang tua tinggi), namun lolos secara aktual karena panitia memprioritaskan IPK yang sangat tinggi (rata-rata 3.51).
2. **Kasus Salah Tolak (216 Pendaftar):** Kelompok ini memiliki beban ekonomi terberat (penghasilan rendah, tanggungan banyak), dengan IPK yang masih di atas batas aman akademik (rata-rata 3.30). Mereka tereliminasi oleh seleksi manual, kemungkinan besar akibat keterbatasan kuota atau bias subjektif panitia.

![Visualisasi Anomali](/app/assets/visualisasi_anomali.png)
*Gambar 2: Sebaran Distribusi Kasus Anomali Keputusan Seleksi*

### 3.5 Validasi Stabilitas Model
Untuk membuktikan validitas ilmiah, dilakukan **Uji Stabilitas Klaster menggunakan metode Split-Sample**. Data dipecah secara acak menjadi dua bagian (50-50), lalu model dilatih kembali secara terpisah. Hasilnya menunjukkan pergeseran koordinat pusat klaster (centroid) maksimal hanya sebesar **5.7%**. Hal ini menunjukkan bahwa klaster kelayakan yang dihasilkan oleh model K-Means sangat stabil, konsisten, dan bukan merupakan pola acak.

### 3.6 Interpretasi Kontribusi Fitur dengan SHAP (Surrogate Model)
Untuk memenuhi ketentuan teknis interpretasi model (SHAP/LIME), proyek ini melatih model *surrogate* (pengganti) berupa Random Forest Classifier untuk meniru keputusan pengelompokan yang dibuat oleh K-Means. Pendekatan ini memungkinkan penggunaan **SHAP (SHapley Additive exPlanations)** untuk mengekstrak signifikansi relatif dari masing-masing fitur.

![SHAP Feature Importance](/app/assets/shap_feature_importance.png)
*Gambar 3: Signifikansi Fitur Pembentuk Klaster Kelayakan menggunakan SHAP*

Berdasarkan analisis nilai SHAP:
* **Jumlah Tanggungan** dan **Penghasilan Orang Tua** merupakan fitur yang paling dominan dalam menentukan klasterisasi pendaftar beasiswa. Hal ini sejalan dengan aspek ekonomi sosial yang menjadi kriteria utama beasiswa sosial.
* **IPK** memiliki pengaruh moderat yang menunjukkan kontribusi akademis yang proporsional, tanpa mendominasi pengelompokan secara ekstrem.
* Fitur keaktifan mahasiswa (seperti **Ikut Organisasi** dan **Ikut UKM**) memiliki nilai SHAP yang sangat rendah, mengonfirmasi bahwa keaktifan non-akademik bukan merupakan penentu prioritas utama kelayakan bantuan ekonomi.

---

# 4. KESIMPULAN DAN REKOMENDASI
### 4.1 Kesimpulan
1. Pemodelan clustering menggunakan algoritma K-Means ($k=2$) berhasil mengelompokkan data pendaftar beasiswa secara objektif menjadi dua tingkat kelayakan (Layak vs Tidak Layak) berdasarkan multikriteria akademik dan ekonomi.
2. Ditemukan tingkat inkonsistensi keputusan seleksi manual historis sebesar **27.4% (286 pendaftar)** jika dibandingkan dengan pengelompokan objektif model.
3. Kasus anomali didominasi oleh kasus **Salah Tolak (216 pendaftar)** dibandingkan kasus **Salah Terima (70 pendaftar)**, yang menunjukkan bahwa sistem seleksi manual historis cenderung mengabaikan pendaftar ekonomi lemah demi mengejar pendaftar ber-IPK sangat tinggi, atau terhambat kuota.

### 4.2 Rekomendasi
1. **Penerapan Sistem Pendukung Keputusan (DSS):** Kampus disarankan mengintegrasikan model K-Means ini sebagai penyaring tahap awal (*automated pre-screening*) pendaftar beasiswa untuk mendeteksi pendaftar yang secara data sangat diprioritaskan.
2. **Penyeimbangan Bobot Seleksi:** Mengkaji ulang kriteria pembobotan panitia seleksi agar aspek sosio-ekonomi dan aspek akademik dinilai secara berimbang dan proporsional.
3. **Audit Keputusan Berkala:** Memanfaatkan daftar hasil ekstraksi anomali untuk melakukan verifikasi lapangan (*field audit*) guna meminimalisir salah sasaran alokasi dana bantuan pendidikan di masa depan.

---

# 5. REFERENSI
1. Han, J., Kamber, M., & Jian, P. (2012). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
4. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224-227.
