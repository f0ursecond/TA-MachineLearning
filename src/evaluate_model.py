# src/evaluate_model.py
import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix, classification_report
from tabulate import tabulate

# Pastikan output menggunakan UTF-8 untuk console Windows
sys.stdout.reconfigure(encoding='utf-8')

# Menambahkan direktori src ke sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def evaluate_best_model(data_path='data/raw/dataset.csv', models_dir='models'):
    """
    Memuat model terbaik dan preprocessor, melakukan evaluasi eksternal,
    dan menampilkan perbandingan detail serta metrik performa.
    """
    print("📈 Memulai Evaluasi Model Terbaik...")
    
    # Load Preprocessor
    prep_path = os.path.join(models_dir, 'preprocessing.pkl')
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor tidak ditemukan di {prep_path}! Silakan jalankan train_model.py terlebih dahulu.")
        
    with open(prep_path, 'rb') as f:
        prep_data = pickle.load(f)
        
    le_dict = prep_data['le_dict']
    scaler = prep_data['scaler']
    fitur_kolom = prep_data['fitur_kolom']
    
    # Load Model K-Means Terbaik
    model_path = os.path.join(models_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di {model_path}! Silakan jalankan train_model.py terlebih dahulu.")
        
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
        
    # Read and clean data
    from data_preprocessing import clean_data, encode_features
    df_clean = clean_data(data_path)
    
    # Encode & Scale
    df_fitur = df_clean[fitur_kolom].copy()
    
    # Encode menggunakan LabelEncoder yang telah difit saat training
    df_encoded = df_fitur.copy()
    for col in df_encoded.columns:
        if col in le_dict:
            le = le_dict[col]
            # Handle unseen categories if any
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            df_encoded[col] = df_encoded[col].apply(lambda x: mapping.get(str(x), 0))
            
    df_scaled = pd.DataFrame(
        scaler.transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index
    )
    
    # Predict clusters
    cluster_labels = kmeans.predict(df_scaled)
    df_clean['Cluster_KMeans'] = cluster_labels
    
    # Metrics
    silhouette_kmeans = silhouette_score(df_scaled, cluster_labels)
    db_kmeans = davies_bouldin_score(df_scaled, cluster_labels)
    
    print("\n" + "="*50)
    print("METRIK EVALUASI INTERNAL CLUSTERING (K-Means)")
    print("="*50)
    print(f"Silhouette Score:      {silhouette_kmeans:.6f}  (Mendekati 1 semakin padat)")
    print(f"Davies-Bouldin Index:  {db_kmeans:.6f}  (Mendekati 0 semakin terpisah)")
    print("="*50)
    
    # Crosstab Comparison
    df_clean['Cluster_KMeans_Label'] = df_clean['Cluster_KMeans'].map({0: 'Cluster 0 (Layak)', 1: 'Cluster 1 (Tidak Layak)'})
    df_clean['Ground_Truth_Label'] = df_clean['Ground_Truth'].map({1: 'Terima Beasiswa', 0: 'Tidak Terima'})
    
    crosstab = pd.crosstab(df_clean['Ground_Truth_Label'], df_clean['Cluster_KMeans_Label'], margins=True)
    
    print("\n" + "="*50)
    print("TABEL KOMPARASI: GROUND TRUTH VS K-MEANS CLUSTER")
    print("="*50)
    print(tabulate(crosstab, headers='keys', tablefmt='psql'))
    print("="*50)
    
    # Print Classification Report (mengasumsikan Cluster 0 adalah Layak / Terima)
    # Mapping Cluster: 0 -> 1 (Terima), 1 -> 0 (Tidak)
    y_pred_mapped = 1 - cluster_labels  # 0 menjadi 1, 1 menjadi 0
    y_true = df_clean['Ground_Truth']
    
    print("\n" + "="*50)
    print("LAPORAN KLASIFIKASI (MENGASUMSIKAN CLUSTER 0 = LAYAK/TERIMA)")
    print("="*50)
    print(classification_report(y_true, y_pred_mapped, target_names=['Tidak Terima', 'Terima Beasiswa']))
    print("="*50)

if __name__ == '__main__':
    evaluate_best_model()
