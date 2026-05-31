import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt

# Memasukkan token otentikasi secara langsung di dalam kode agar otomatis login
os.environ["DAGSHUB_CLIENT_TOKEN"] = "31bda968114976ca672d715116d12f44a956d460"

# 3.1. Matikan Autolog
mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)

# 3.2 Inisialisasi koneksi online ke repositori DagsHub
dagshub.init(repo_owner='Dedy-art', repo_name='predictive-maintenance-mlflow', mlflow=True)

def main():
  # Load dataset yang sudah di preprocess
  data_path = 'Membangun_model/predictive_maintenance_preprocessed.csv'
  if not os.path.exists(data_path):
    print(f"Data tidak ditemukan di {data_path}!")
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
  with mlflow.start_run(run_name="Baseline_Random_Forest"):
    print("Sedang melatih model Random Forest")

    # Inisialisasi model (Menggunakan default hyperparameter untuk kriteria Basic)
    model = RandomForestClassifier(random_state=42)

    # Latih model
    model.fit(X_train, y_train)

    # Prediksi da evaluasi
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Pelatihan Selesai! Akurasi Model: {akurasi:.4f}")

    # Manual Logging parameter & metrik
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", model.n_estimators)

    # Mencatat metrik akurasi secara manual
    mlflow.log_metric("accuracy_score", akurasi)

    # Manual Logging artefak tambahan (Minimal 2 Artefak)
    # Artefak 1: Menyimpan file teks hasil Classification Report
    report_path = "Membangun_model/Classification_Report.txt"
    with open(report_path, "w") as f:
      f.write(report)
    mlflow.log_artifact(report_path)

    # Artefak 2: Menyimpan gambar grafik confusion Metrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plot_path = "Membangun_model/Confusion_Matrix.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(plot_path)

    # Save dan log model Sciki-Learn ke MLflow
    mlflow.sklearn.log_model(model, "baseline_model")

    print("Semua matrik, parameter, dan 2 artefak sukses diupload ke DagsHub secara murni manual")


if __name__ == "__main__":
  main()
