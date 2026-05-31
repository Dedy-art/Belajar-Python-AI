import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import dagshub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=10)
    args = parser.parse_args()

    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    dagshub_repo = os.environ.get("DAGSHUB_REPO_NAME", "predictive-maintenance-mlflow")

    if dagshub_username and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Tracking dialihkan ke DagsHub: {tracking_uri}")
    else:
        print("Peringatan: Menggunakan tracking lokal.")

    mlflow.set_experiment("Predictive_Maintenance_CI")
    mlflow.sklearn.autolog(disable=True)

    # PERBAIKAN: Menyesuaikan nama dataset yang dipindahkan di langkah 4
    dataset_path = "MLProject/predictive_maintenance_preprocessed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "predictive_maintenance_preprocessed.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} tidak ditemukan!")
        return

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=['Machine failure'], errors='ignore')
    y = df['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", accuracy)

        report_text = classification_report(y_test, y_pred)
        with open("CI_Classification_Report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("CI_Classification_Report.txt")

        plt.figure(figsize=(6,4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig("CI_Confusion_Matrix.png", bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("CI_Confusion_Matrix.png")

        # Log Model ke MLflow Server
        mlflow.sklearn.log_model(model, "model")

        # PERBAIKAN TARGET ADVANCED: Save lokal khusus agar aman di-build oleh Docker tanpa folder mlruns
        mlflow.sklearn.save_model(model, "fixed_local_model")
        print(f"Training Selesai! Akurasi: {accuracy:.4f}. Model aman tersimpan lokal.")

if __name__ == "__main__":
    main()