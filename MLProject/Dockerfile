# Menggunakan base image Python resmi yang ringan
FROM python:3.10-slim

# Mengatur lokasi kerja di dalam container
WORKDIR /app

# Menyalin file requirements atau model ke dalam container
COPY . /app

# Menginstall library utama yang dibutuhkan untuk menjalankan model
RUN pip install --no-cache-dir mlflow pandas numpy scikit-learn

# Perintah default saat container dijalankan
CMD ["python", "modelling.py"]