# src/train_model.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Pastikan output menggunakan UTF-8 untuk console Windows
sys.stdout.reconfigure(encoding='utf-8')

# Menambahkan direktori src ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from data_preprocessing import clean_data, encode_features, scale_features
import utils


def run_pipeline(raw_data_path='data/raw/dataset.csv', models_dir='models', processed_dir='data/processed'):
    """
    Menjalankan pipeline Machine Learning lengkap:
    1. Preprocessing & Pembersihan data
    2. Pemodelan & Evaluasi (K-Means, Agglomerative, DBSCAN)
    3. Ekspor model terbaik & preprocessor
    4. Identifikasi anomali beasiswa
    5. Penyimpanan visualisasi analisis
    """
    print("🚀 Memulai Pipeline Pelatihan Model...")
    
    # Pastikan struktur folder ada
    utils.create_directory_structure()
    
    # 1. Cleaning data
    if not os.path.exists(raw_data_path):
        # Jika file belum dipindah, coba cari di root folder
        if os.path.exists('dataset.csv'):
            print(f"Memindahkan dataset.csv ke {raw_data_path}...")
            os.rename('dataset.csv', raw_data_path)
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di {raw_data_path} atau root!")
            
    df_clean = clean_data(raw_data_path)
    print(f"✅ Data dibersihkan. Jumlah data: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom.")
    
    # Tentukan fitur yang digunakan untuk clustering
    fitur_kolom = [col for col in df_clean.columns 
                   if col not in ['No', 'Nama Lengkap', 'Asal Sekolah', 'Status Beasiswa', 'Ground_Truth', 'Prodi']]
    print(f"📋 Fitur clustering ({len(fitur_kolom)}): {fitur_kolom}")
    
    # 2. Encoding & Scaling
    df_fitur = df_clean[fitur_kolom].copy()
    df_encoded, le_dict = encode_features(df_fitur)
    df_scaled, scaler = scale_features(df_encoded)
    
    # Simpan preprocessor (LabelEncoder dict & StandardScaler)
    os.makedirs(models_dir, exist_ok=True)
    prep_path = os.path.join(models_dir, 'preprocessing.pkl')
    with open(prep_path, 'wb') as f:
        pickle.dump({
            'le_dict': le_dict,
            'scaler': scaler,
            'fitur_kolom': fitur_kolom
        }, f)
    print(f"💾 Preprocessor disimpan ke: {prep_path}")
    
    # 3. Tuning & Eksperimen Elbow / Silhouette
    print("⏳ Melakukan WCSS & Silhouette tuning...")
    utils.plot_elbow_method(df_scaled)
    
    # 4. Melatih Model
    print("🤖 Melatih model K-Means, Agglomerative, & DBSCAN...")
    
    # K-Means (Model Utama)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    cluster_labels_kmeans = kmeans.fit_predict(df_scaled)
    df_clean['Cluster_KMeans'] = cluster_labels_kmeans
    
    silhouette_kmeans = silhouette_score(df_scaled, cluster_labels_kmeans)
    db_kmeans = davies_bouldin_score(df_scaled, cluster_labels_kmeans)
    
    # Simpan Model K-Means Terbaik
    model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"💾 Model K-Means terbaik disimpan ke: {model_path}")
    
    # Agglomerative
    agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
    cluster_labels_agglo = agglo.fit_predict(df_scaled)
    df_clean['Cluster_Agglomerative'] = cluster_labels_agglo
    
    # Train proxy classifier for Agglomerative prediction on new data
    from sklearn.neighbors import KNeighborsClassifier
    agglo_proxy = KNeighborsClassifier(n_neighbors=3)
    agglo_proxy.fit(df_scaled, cluster_labels_agglo)
    
    agglo_model_path = os.path.join(models_dir, 'agglo_model.pkl')
    with open(agglo_model_path, 'wb') as f:
        pickle.dump(agglo_proxy, f)
    print(f"💾 Proxy model Agglomerative disimpan ke: {agglo_model_path}")
    
    silhouette_agglo = silhouette_score(df_scaled, cluster_labels_agglo)
    db_agglo = davies_bouldin_score(df_scaled, cluster_labels_agglo)
    
    # DBSCAN
    dbscan = DBSCAN(eps=2.5, min_samples=5)
    cluster_labels_dbscan = dbscan.fit_predict(df_scaled)
    df_clean['Cluster_DBSCAN'] = cluster_labels_dbscan
    
    valid_mask = cluster_labels_dbscan != -1
    if valid_mask.sum() > 0 and len(set(cluster_labels_dbscan[valid_mask])) > 1:
        silhouette_dbscan = silhouette_score(df_scaled[valid_mask], cluster_labels_dbscan[valid_mask])
    else:
        silhouette_dbscan = 0.0
        
    print("\n" + "="*40)
    print("HASIL PERBANDINGAN EVALUASI CLUSTERING")
    print("="*40)
    print(f"K-Means:       Silhouette = {silhouette_kmeans:.4f}, DB Index = {db_kmeans:.4f}")
    print(f"Agglomerative: Silhouette = {silhouette_agglo:.4f}, DB Index = {db_agglo:.4f}")
    print(f"DBSCAN:        Silhouette = {silhouette_dbscan:.4f}, Noise = {(cluster_labels_dbscan == -1).sum()} data")
    print("="*40 + "\n")
    
    # 5. Membuat Visualisasi Evaluasi & Insight
    print("📊 Membuat visualisasi analisis...")
    utils.plot_heatmap_korelasi(df_clean, fitur_kolom)
    utils.plot_boxplot_perbandingan(df_clean)
    utils.plot_confusion_matrix(df_clean['Ground_Truth'], cluster_labels_kmeans, cluster_labels_agglo)
    utils.plot_scatter_ipk_penghasilan(df_clean)
    utils.plot_shap_interpretation(df_scaled, cluster_labels_kmeans)

    
    # 6. Analisis Anomali
    print("🔍 Mengidentifikasi anomali penerimaan beasiswa...")
    # Tentukan interpretasi kelayakan cluster K-Means
    # Berdasarkan profil cluster:
    # Cluster 0 memiliki rata-rata tanggungan tinggi (3.16) & penghasilan rendah yang tinggi -> LAYAK
    # Cluster 1 memiliki rata-rata tanggungan rendah (2.20) & penghasilan rendah yang rendah -> TIDAK LAYAK
    df_clean['Cluster_KMeans_Label'] = df_clean['Cluster_KMeans'].map({0: 'Layak', 1: 'Tidak Layak'})
    
    # Filter data anomali
    # Anomali 1: Salah Terima (Aktual: Terima, Model: Tidak Layak)
    # Anomali 2: Salah Tolak (Aktual: Tidak, Model: Layak)
    df_anomali = df_clean[
        ((df_clean['Ground_Truth'] == 1) & (df_clean['Cluster_KMeans'] == 1)) |
        ((df_clean['Ground_Truth'] == 0) & (df_clean['Cluster_KMeans'] == 0))
    ].copy()
    
    def get_tipe_anomali(row):
        if row['Ground_Truth'] == 1 and row['Cluster_KMeans'] == 1:
            return 'Salah Terima'
        elif row['Ground_Truth'] == 0 and row['Cluster_KMeans'] == 0:
            return 'Salah Tolak'
        return 'Sesuai'
        
    df_anomali['Tipe_Anomali'] = df_anomali.apply(get_tipe_anomali, axis=1)
    
    # Simpan data anomali
    os.makedirs(processed_dir, exist_ok=True)
    anomali_path = os.path.join(processed_dir, 'data_anomali.csv')
    df_anomali.to_csv(anomali_path, index=False, encoding='cp1252')
    print(f"💾 Data anomali disimpan ke: {anomali_path}")
    
    # Buat plot visualisasi anomali
    utils.plot_visualisasi_anomali(df_anomali, df_clean)
    
    print("\n✅ Pipeline Pelatihan dan Analisis Model Berhasil Diselesaikan!")
    print(f"   • Total Pendaftar: {len(df_clean)}")
    print(f"   • Total Anomali:   {len(df_anomali)} (Salah Terima: {sum(df_anomali['Tipe_Anomali'] == 'Salah Terima')}, Salah Tolak: {sum(df_anomali['Tipe_Anomali'] == 'Salah Tolak')})")

if __name__ == '__main__':
    run_pipeline()
