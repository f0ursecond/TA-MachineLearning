# src/utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def create_directory_structure():
    """
    Membuat struktur folder repositori sesuai format UAS.
    """
    folders = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'models',
        'app/assets',
        'reports'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Struktur direktori repositori berhasil dibuat!")

def save_plot(fig, filename, assets_dir='app/assets'):
    """
    Menyimpan plot ke dalam direktori assets.
    """
    os.makedirs(assets_dir, exist_ok=True)
    path = os.path.join(assets_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Visualisasi disimpan ke: {path}")

def plot_heatmap_korelasi(df, features, assets_dir='app/assets'):
    """
    Membuat dan menyimpan heatmap korelasi.
    """
    df_numeric = df[features].copy()
    for col in df_numeric.columns:
        if not pd.api.types.is_numeric_dtype(df_numeric[col]):
            le = LabelEncoder()
            df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
            
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur")
    save_plot(fig, "heatmap_korelasi.png", assets_dir)

def plot_boxplot_perbandingan(df, assets_dir='app/assets'):
    """
    Membuat dan menyimpan boxplot sebaran kriteria utama.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fitur_visual = ['IPK', 'Tanggungan', 'SKS']
    
    # 3 Boxplot pertama untuk perbandingan berdasarkan keputusan beasiswa aktual
    for idx, fit in enumerate(fitur_visual):
        ax = axes[0, idx]
        sns.boxplot(data=df, x='Status Beasiswa', y=fit, ax=ax, palette='Set2')
        ax.set_title(f"Sebaran {fit} vs Status Beasiswa")
        
    # 3 Boxplot kedua untuk perbandingan berdasarkan Cluster K-Means
    if 'Cluster_KMeans' in df.columns:
        cluster_labels = df['Cluster_KMeans'].map({0: 'Cluster 0 (Layak)', 1: 'Cluster 1 (Tidak Layak)'})
        for idx, fit in enumerate(fitur_visual):
            ax = axes[1, idx]
            sns.boxplot(data=df, x=cluster_labels, y=fit, ax=ax, palette='Pastel1')
            ax.set_title(f"Sebaran {fit} vs Cluster K-Means")
            
    plt.tight_layout()
    save_plot(fig, "boxplot_perbandingan.png", assets_dir)

def plot_elbow_method(df_scaled, k_range=range(2, 11), assets_dir='app/assets'):
    """
    Menghitung WCSS dan Silhouette Score, lalu membuat plot Elbow Method.
    """
    wcss = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(df_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, labels))
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot WCSS
    color = 'tab:red'
    ax1.set_xlabel('Jumlah Cluster (k)')
    ax1.set_ylabel('WCSS (Inertia)', color=color)
    ax1.plot(k_range, wcss, 'o-', color=color, label='WCSS')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot Silhouette Score
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, 's--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Metode Elbow dan Silhouette Score untuk Penentuan k Optimal')
    fig.tight_layout()
    save_plot(fig, "elbow_method.png", assets_dir)
    
    return wcss, silhouette_scores

def plot_confusion_matrix(y_true, y_pred_kmeans, y_pred_agglo=None, assets_dir='app/assets'):
    """
    Membuat dan menyimpan confusion matrix evaluasi clustering terhadap Ground Truth.
    """
    cm_kmeans = confusion_matrix(y_true, y_pred_kmeans)
    
    if y_pred_agglo is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot K-Means CM
        sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Cluster 0', 'Cluster 1'],
                    yticklabels=['Tidak Terima', 'Terima Beasiswa'])
        axes[0].set_title('K-Means Cluster vs Ground Truth')
        axes[0].set_xlabel('Prediksi Cluster')
        axes[0].set_ylabel('Aktual (Ground Truth)')
        
        # Plot Agglomerative CM
        cm_agglo = confusion_matrix(y_true, y_pred_agglo)
        sns.heatmap(cm_agglo, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=['Cluster 0', 'Cluster 1'],
                    yticklabels=['Tidak Terima', 'Terima Beasiswa'])
        axes[1].set_title('Agglomerative Cluster vs Ground Truth')
        axes[1].set_xlabel('Prediksi Cluster')
        axes[1].set_ylabel('Aktual (Ground Truth)')
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Cluster 0', 'Cluster 1'],
                    yticklabels=['Tidak Terima', 'Terima Beasiswa'])
        ax.set_title('K-Means Cluster vs Ground Truth')
        ax.set_xlabel('Prediksi Cluster')
        ax.set_ylabel('Aktual (Ground Truth)')
        
    save_plot(fig, "confusion_matrix.png", assets_dir)

def plot_scatter_ipk_penghasilan(df_clean, assets_dir='app/assets'):
    """
    Membuat dan menyimpan scatter plot IPK vs Penghasilan berdasarkan kategori kesesuaian seleksi.
    """
    df_scatter = df_clean.copy()
    
    # Map Penghasilan ke nilai numerik untuk plotting
    penghasilan_map = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    df_scatter['Penghasilan_num'] = df_scatter['Penghasilan'].map(penghasilan_map)
    
    # Tambah noise kecil agar titik scatter tidak bertumpuk sempurna di nilai integer penghasilan
    np.random.seed(42)
    df_scatter['Penghasilan_jittered'] = df_scatter['Penghasilan_num'] + np.random.uniform(-0.15, 0.15, size=len(df_scatter))
    
    # Tentukan kategori kesesuaian seleksi
    def get_kategori(row):
        if row['Ground_Truth'] == 1 and row['Cluster_KMeans'] == 0:
            return 'Benar: Terima'
        elif row['Ground_Truth'] == 0 and row['Cluster_KMeans'] == 1:
            return 'Benar: Tolak'
        elif row['Ground_Truth'] == 1 and row['Cluster_KMeans'] == 1:
            return 'Salah Terima'
        elif row['Ground_Truth'] == 0 and row['Cluster_KMeans'] == 0:
            return 'Salah Tolak'
        return 'Tidak Diketahui'
        
    df_scatter['Kategori'] = df_scatter.apply(get_kategori, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Benar: Tolak': '#2ca02c', 'Benar: Terima': '#1f77b4', 'Salah Tolak': '#9467bd', 'Salah Terima': '#d62728'}
    
    sns.scatterplot(
        data=df_scatter,
        x='IPK',
        y='Penghasilan_jittered',
        hue='Kategori',
        palette=colors,
        alpha=0.7,
        ax=ax
    )
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Rendah', 'Sedang', 'Tinggi'])
    ax.set_ylabel('Penghasilan Orang Tua')
    ax.set_xlabel('IPK Pendaftar')
    ax.set_title('Kesesuaian Keputusan Seleksi Aktual vs Analisis Clustering (IPK vs Penghasilan)')
    
    save_plot(fig, "scatter_ipk_penghasilan.png", assets_dir)

def plot_visualisasi_anomali(df_anomali, df_clean, assets_dir='app/assets'):
    """
    Membuat dan menyimpan visualisasi distribusi anomali.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Kategori Kesesuaian (Pie/Bar)
    ax1 = axes[0, 0]
    def get_kategori(row):
        if row['Ground_Truth'] == 1 and row['Cluster_KMeans'] == 0:
            return 'Benar: Terima'
        elif row['Ground_Truth'] == 0 and row['Cluster_KMeans'] == 1:
            return 'Benar: Tolak'
        elif row['Ground_Truth'] == 1 and row['Cluster_KMeans'] == 1:
            return 'Salah Terima'
        elif row['Ground_Truth'] == 0 and row['Cluster_KMeans'] == 0:
            return 'Salah Tolak'
        return 'Lainnya'
    
    df_temp = df_clean.copy()
    df_temp['Kategori'] = df_temp.apply(get_kategori, axis=1)
    kategori_counts = df_temp['Kategori'].value_counts()
    
    sns.barplot(x=kategori_counts.index, y=kategori_counts.values, ax=ax1, palette='Set2')
    ax1.set_title("Proporsi Keputusan Seleksi (Clustering vs Ground Truth)")
    ax1.set_ylabel("Jumlah Pendaftar")
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                    
    # Plot 2: Perbandingan Jumlah Tipe Anomali
    ax2 = axes[0, 1]
    if len(df_anomali) > 0:
        anomali_counts = df_anomali['Tipe_Anomali'].value_counts()
        sns.barplot(x=anomali_counts.index, y=anomali_counts.values, ax=ax2, palette='Set1')
        ax2.set_title("Perbandingan Kasus Anomali")
        ax2.set_ylabel("Jumlah Kasus")
        for p in ax2.patches:
            ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    else:
        ax2.text(0.5, 0.5, "Tidak ada anomali terdeteksi", ha='center', va='center')
        
    # Plot 3: Distribusi IPK pada data anomali
    ax3 = axes[1, 0]
    if len(df_anomali) > 0:
        sns.histplot(data=df_anomali, x='IPK', hue='Tipe_Anomali', element='step', stat='density', common_norm=False, ax=ax3, palette='Set1')
        ax3.set_title("Distribusi IPK pada Kelompok Anomali")
    else:
        ax3.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        
    # Plot 4: Distribusi Tanggungan pada data anomali
    ax4 = axes[1, 1]
    if len(df_anomali) > 0:
        sns.boxplot(data=df_anomali, x='Tipe_Anomali', y='Tanggungan', ax=ax4, palette='Set1')
        ax4.set_title("Jumlah Tanggungan pada Kelompok Anomali")
    else:
        ax4.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        
    plt.tight_layout()
    save_plot(fig, "visualisasi_anomali.png", assets_dir)
