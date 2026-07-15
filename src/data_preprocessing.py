# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(file_path):
    """
    Membaca dataset, menghapus kolom kosong Unnamed, menghapus duplikat,
    dan melakukan imputasi missing values persis seperti di Jupyter Notebook.
    """
    # Membaca dataset dengan encoding cp1252 (ANSI)
    df = pd.read_csv(file_path, encoding='cp1252')
    
    # Salin data untuk dibersihkan
    df_clean = df.copy()
    
    # Hapus kolom yang seluruh nilainya NaN (kolom Unnamed: 15 s/d Unnamed: 25)
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Bersihkan whitespace pada nama kolom
    df_clean.columns = df_clean.columns.str.strip()
    
    # Hapus duplikat
    df_clean = df_clean.drop_duplicates()
    
    # Imputasi missing values: median untuk numerik, mode untuk kategorikal
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
    # Map Status Beasiswa ke Ground Truth biner (1 untuk Terima, 0 untuk Tidak)
    if 'Status Beasiswa' in df_clean.columns:
        df_clean['Ground_Truth'] = df_clean['Status Beasiswa'].map({'Terima': 1, 'Tidak': 0})
        df_clean['Ground_Truth'] = df_clean['Ground_Truth'].fillna(0).astype(int)
        
    return df_clean

def encode_features(df_fitur):
    """
    Melakukan LabelEncoding untuk kolom kategorikal.
    Mengembalikan dataframe ter-encode dan dictionary LabelEncoder.
    """
    df_encoded = df_fitur.copy()
    le_dict = {}
    
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
            
    return df_encoded, le_dict

def scale_features(df_encoded):
    """
    Melakukan standardisasi data numerik menggunakan StandardScaler.
    Mengembalikan dataframe ter-scaling dan StandardScaler object.
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index
    )
    return df_scaled, scaler
