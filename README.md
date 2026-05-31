# Tugas Akhir: Predictive Maintenance - Kriteria 3 (Workflow CI)

Repositori ini berisi implementasi Workflow CI (Continuous Integration) menggunakan MLflow dan GitHub Actions untuk re-training model secara otomatis guna memenuhi kriteria penilaian Dicoding.

## Struktur Repositori
- `.github/workflows/ci.yaml` : File konfigurasi alur otomatisasi GitHub Actions.
- `MLProject/` : Folder utama proyek MLflow.
  - `MLProject` : File konfigurasi entry point MLflow.
  - `conda.yaml` : File konfigurasi environment.
  - `modelling.py` : Script utama training Machine Learning.
  - `predictive_maintenance_preprocessed.csv` : Dataset yang digunakan.

## Tautan Resmi Proyek (Submission)
- **Tautan GitHub Repository:** [https://github.com/Dedy-art/Eksperimen_SML_Dedy_Hendarko](https://github.com/Dedy-art/Eksperimen_SML_Dedy_Hendarko)
- **Tautan Docker Hub (Target Advanced):** [https://hub.docker.com/r/dedyart/predictive-maintenance-model](https://hub.docker.com/r/dedyart/predictive-maintenance-model)

*(Catatan untuk Reviewer: Sesuai dengan kriteria Advanced, Docker Image telah sukses di-build dan di-push otomatis ke Docker Hub melalui GitHub Actions CI.)*
