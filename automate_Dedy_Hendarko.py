import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
  """Memuat dataset dari jalur file"""
  return pd.read_csv(file_path)

def preprocess_data(df):
  """Melakukan otomasi seluruh tahapan preprocessing"""
  df_clean = df.copy()

  # 1. Hapus kolom identitas
  columns_to_drop = ['UDI', 'Product ID']
  df_clean = df_clean.drop(columns=columns_to_drop, axis=1, errors='ignore')

  # 2. One-Hot Encoding pada kolom 'Type'
  if 'Type' in df_clean.columns:
    df_clean = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)
    # Pastikan tipenya angka bulat agar aman di GitHub Actions
    df_clean = df_clean.astype({col: 'int64' for col in df_clean.select_dtypes(include='bool').columns})

  # 3. Pisahkan Fitur dan Target
  if 'Machine failure in df_clean.columns:
    X = df_clean.drop(columns=['Machine failure'])
    y = df_clean['Machine failure']
  else:
    X = df_clean
    y = None

  # 4. Standarisasi Fiture Numerik
  numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
  scaler = StandardScaler()
  X_scaled = X.copy()
  X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

  # 5. Gabungkan kembali jika ada target
  if y is not None:
    df_final =X_scaled.copy()
    df_final['Machine failure'] = y
  else:
    df_final = X_scaled

  return df_final

if __name__ == "__main__":
  import os

  # Tentukan jalur file (ini disesuaikan dengan nama file asli Dedy_Hendarko)
  input_file = 'namadataset_raw/predictive_maintenance_ai4i2020.csv'
  output_dir = 'preprocessing/namadataset_preprocessing'
  output_file = os.path.join(output_dir, 'predictive_maintenance_preprocessing.csv')

  # Jalankan jika file input ada (untuk testing local/workflow)
  if os.path.exists(input_file):
    print("Memulai otomatisasi preprocessing...")
    raw_data = load_data(input_file)
    clean_data = preprocess_data(raw_data)

    os.makedirs(output_dir, exist_ok=True)
    clean_data.to_csv(output_file, index=False)
    print(f"Sukses! Data Siap latih & disimpan di: {output_file}")
  else:
    print(f"File {input_file} tidak ditemukan. Otomatisasi siap digunakan dalam workflow.")