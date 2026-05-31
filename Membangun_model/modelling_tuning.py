import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import shutil

# 6.1 Memasukkan token otentikasi secara langsung di dalam kode agar otomatis login
os.environ["DAGSHUB_CLIENT_TOKEN"] = "31bda968114976ca672d715116d12f44a956d460"

# 6.2 Matikan Autolog
mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)

# 6.3 Inisialisasi koneksi online ke repositori DagsHub
dagshub.init(repo_owner='Dedy-art', repo_name='predictive-maintenance-mlflow', mlflow=True)

def main():
  # Load dataset yang sudah di preprocess
  data_path = 'Membangun_model/predictive_maintenance_preprocessed.csv'
  if not os.path.exists(data_path):
    print(f"Error: Data tidak ditemukan di {data_path}!")
    return

  df = pd.read_csv(data_path)

  # Pisahkan Fitur (X) dan Target (y)
  # Target utama berdasarkan EDA kemarin adalah 'Machine failure'
  X = df.drop(columns=['Machine failure'])
  y = df['Machine failure']

  # Split menjadi data Traning & Testing (80% train, 20% test)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # Set nama Eksperimen di MLflow DagsHub
  mlflow.set_experiment("Eksperimen Predictive Maintenance")

  # Memulai run MLflow
  with mlflow.start_run(run_name="Tuned_Random_Forest_Manual"):
    print("Memulai proses Hyperparameter Tuning dengan GridSearchCV...")

    # Definisikan kombinasi parameter yang akan diuji
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
        }

    # Inisialisasi model (Menggunakan default hyperparameter untuk kriteria Basic)
    rf_base = RandomForestClassifier(random_state=42)

    # Jalankan GridSearchCV (CV=3 agar prosesnya cepat di Colab)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Ambil model terbaik hasil tuning
    best_model = grid_search.best_estimator_

    # Prediksi menggunakan model terbaik
    y_pred = best_model.predict(X_test)

    # Hitung metrik evaluasi hasil tuning
    akurasi = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Tuning Selesai! Akurasi Model Terbaik: {akurasi:.4f}")
    print(f"Parameter Terbaik: {grid_search.best_params_}")

    # Manual Logging parameter & metrik
    mlflow.log_param("model_type", "Tuned_RandomForestClassifier")
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])

    # Mencatat metrik akurasi secara manual
    mlflow.log_metric("accuracy_score", akurasi)

    # Manual Logging artefak tambahan (Minimal 2 Artefak)
    # Artefak 1: Menyimpan file teks hasil Tuning Classification Report
    report_path = "Membangun_model/Tuning_Classification_Report.txt"
    with open(report_path, "w") as f:
      f.write(report)
    mlflow.log_artifact(report_path)

    # Artefak 2: Menyimpan gambar grafik Tuning confusion Metrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plot_path = "Membangun_model/Tuning_Confusion_Matrix.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(plot_path)

    # Save dan log model Sciki-Learn ke MLflow
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="tuned_model")

    print("Semua matrik, parameter, dan 2 artefak tuning sukses diupload ke DagsHub secara murni manual")


if __name__ == "__main__":
  main()
