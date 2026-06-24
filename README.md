## Dashboard Prediksi TWP90

Dashboard ini merupakan hasil implementasi dari model prediksi **TWP90 (%)** yang telah dibangun menggunakan pendekatan **Hybrid SARIMAX + XGBoost Residual**. Model ini dikembangkan untuk membantu menampilkan hasil prediksi TWP90 secara interaktif berdasarkan data historis dan input variabel eksternal yang diberikan oleh pengguna.

Repository proyek dapat diakses melalui:

**GitHub Repository:**
https://github.com/geeefidd-wq/dashboardprediksi_TWP90/tree/main

Dashboard yang telah dideploy dapat diakses melalui:

**Live Dashboard Streamlit:**
https://dashboardprediksitwp90.streamlit.app/

---

## Deskripsi Singkat

Dashboard ini dibuat untuk menampilkan hasil pemodelan prediksi TWP90 dengan skema **walk-forward one-step-ahead forecasting**. Artinya, prediksi dilakukan secara berurutan satu bulan ke depan, kemudian data aktual pada bulan sebelumnya digunakan kembali untuk memperbarui proses prediksi bulan berikutnya.

Model yang digunakan bukan hanya memprediksi nilai TWP90 secara langsung, tetapi menggabungkan dua pendekatan:

1. **SARIMAX** digunakan untuk menangkap pola time series, tren, musiman, dan pengaruh variabel eksternal.
2. **XGBoost** digunakan untuk mempelajari residual atau selisih kesalahan dari model SARIMAX pada skala log.

Dengan pendekatan hybrid ini, model diharapkan dapat menghasilkan prediksi yang lebih stabil karena SARIMAX menangani pola runtun waktu, sedangkan XGBoost membantu memperbaiki sisa kesalahan prediksi dari SARIMAX.

---

## Hasil Pemodelan

Berdasarkan proses evaluasi model, skema yang digunakan adalah **walk-forward one-step-ahead** dengan window expanding. Model terbaik yang digunakan pada dashboard adalah:

* **Model utama:** Hybrid SARIMAX + XGBoost Residual
* **Target prediksi:** TWP90 (%)
* **Horizon prediksi:** 1 bulan ke depan
* **Skema validasi:** Time Series Cross-Validation dan Walk-Forward Forecasting
* **Target XGBoost:** Residual SARIMAX pada skala log

Ringkasan hasil evaluasi model pada data test:

| Model                                            |     RMSE |      MAE |     MAPE |
| ------------------------------------------------ | -------: | -------: | -------: |
| SARIMAX One-Step-Ahead                           | 0.005175 | 0.003388 | 9.992254 |
| Hybrid SARIMAX + XGBoost Residual One-Step-Ahead | 0.005119 | 0.003346 | 9.928552 |

Dari hasil tersebut, model hybrid menunjukkan performa yang sedikit lebih baik dibandingkan SARIMAX tunggal karena mampu menurunkan nilai RMSE, MAE, dan MAPE pada data test.

---

## Logika Prediksi pada Dashboard

Pada dashboard ini, pengguna tidak hanya memilih periode input, tetapi memilih **periode TWP90 yang ingin diprediksi**. Setelah periode target dipilih, sistem akan menentukan data input apa saja yang dibutuhkan sampai bulan sebelum target.

Contoh logika prediksi:

* Jika data historis terakhir adalah **Desember 2025**
* Pengguna memilih target **Februari 2026**
* Maka pengguna perlu mengisi data input untuk **Januari 2026**
* Sistem akan menampilkan hasil prediksi untuk **Januari 2026 dan Februari 2026**

Contoh lainnya:

| Target Prediksi | Input yang Dibutuhkan             | Output yang Ditampilkan                           |
| --------------- | --------------------------------- | ------------------------------------------------- |
| Februari 2026   | Januari 2026                      | Prediksi Januari dan Februari 2026                |
| Maret 2026      | Januari dan Februari 2026         | Prediksi Januari, Februari, dan Maret 2026        |
| April 2026      | Januari, Februari, dan Maret 2026 | Prediksi Januari, Februari, Maret, dan April 2026 |

Dengan alur ini, dashboard tetap mengikuti prinsip **one-step-ahead forecasting**, sehingga proses prediksi tidak langsung melompati bulan target tanpa memperbarui data historis sebelumnya.

---

## Alasan TWP90 Aktual Ikut Diinput

Pada dashboard, pengguna diminta mengisi nilai TWP90 aktual untuk bulan-bulan sebelum periode target. Hal ini bukan berarti dashboard meminta nilai TWP90 target yang ingin diprediksi.

Nilai TWP90 aktual hanya digunakan untuk bulan sebelum target agar sistem dapat:

1. memperbarui state model SARIMAX secara berurutan;
2. menghitung residual aktual SARIMAX pada bulan input;
3. memperbarui histori residual untuk XGBoost;
4. menjaga skema prediksi tetap sesuai dengan metode one-step-ahead.

Dengan demikian, nilai TWP90 aktual tidak digunakan untuk membocorkan nilai target. Bulan target tetap diprediksi oleh model.

---

## Fitur Dashboard

Dashboard menyediakan beberapa fitur utama, yaitu:

* Menampilkan ringkasan hasil pemodelan TWP90.
* Menampilkan grafik data aktual dan hasil prediksi.
* Menampilkan hasil evaluasi model.
* Melakukan prediksi TWP90 berdasarkan periode target yang dipilih pengguna.
* Menggunakan input variabel eksternal untuk membentuk fitur prediksi.
* Menghitung prediksi secara berurutan sesuai skema walk-forward.
* Membaca artifact model dari folder `model_artifacts/`.

---

## Variabel Input Dashboard

Dashboard membutuhkan data historis dan input variabel eksternal berikut:

* `Month`
* `TWP90 (%)`
* `Outstanding Pinjaman (miliar RP)`
* `BI-7Day-RR`
* `Inflasi`
* `PDB (miliar Rp)`
* `Pertumbuhan Outstanding (YoY% atau MoM%)`
* `Indeks Keyakinan Konsumen (IKK)`
* `Nilai Tukar Rupiah terhadap USD`

Kolom persen diisi dalam bentuk angka persen asli, bukan bentuk desimal internal model.

Contoh pengisian:

| Variabel                       | Contoh Input |
| ------------------------------ | -----------: |
| TWP90 4,32%                    |         4.32 |
| Inflasi 2,92%                  |         2.92 |
| BI Rate 5,75%                  |         5.75 |
| Pertumbuhan Outstanding 18,03% |        18.03 |

Nilai `0` dianggap sebagai data yang belum diisi. Oleh karena itu, prediksi tidak akan ditampilkan apabila masih terdapat input bernilai `0` atau kosong.

---

## Struktur Artifact Model

Dashboard membaca file artifact dari folder `model_artifacts/`. Struktur folder yang disarankan adalah sebagai berikut:

```text
project_dashboard/
├── app.py
├── pipeline_export_artifacts.py
├── pipeline_export_artifacts.ipynb
├── README_DEPLOYMENT.md
└── model_artifacts/
    ├── hybrid_twp90_model.joblib
    ├── preprocessing_config.json
    └── raw_history.csv
```

File utama yang digunakan:

| File                           | Fungsi                                                      |
| ------------------------------ | ----------------------------------------------------------- |
| `app.py`                       | File utama dashboard Streamlit                              |
| `hybrid_twp90_model.joblib`    | Artifact model hybrid yang sudah dilatih                    |
| `preprocessing_config.json`    | Konfigurasi preprocessing dan metadata model                |
| `raw_history.csv`              | Data historis mentah untuk membentuk fitur lag dan residual |
| `pipeline_export_artifacts.py` | Script sinkronisasi artifact deployment                     |

---

## Pentingnya `raw_history.csv`

File `raw_history.csv` diperlukan karena model menggunakan fitur berbasis lag dan differencing, seperti:

* `BI-7Day-RR_lag1`
* `Inflasi_lag1`
* `IKK_lag6`
* `Pertumbuhan Outstanding_lag12`
* fitur residual SARIMAX

Oleh karena itu, dashboard membutuhkan data historis mentah agar fitur lag dapat dibentuk dengan benar. Jika file `raw_history.csv` tidak tersedia, dashboard masih dapat membaca model dan hasil evaluasi dari artifact, tetapi menu prediksi tidak dapat berjalan secara lengkap.

---

## Cara Menjalankan Dashboard

Install library yang dibutuhkan:

```bash
pip install streamlit pandas numpy plotly joblib pmdarima xgboost scikit-learn statsmodels
```

Jalankan dashboard:

```bash
streamlit run app.py
```

Dashboard juga dapat membaca `hybrid_twp90_model.joblib` dan `preprocessing_config.json` apabila kedua file tersebut diletakkan satu folder dengan `app.py`.

---

## Cara Menjalankan Pipeline Artifact

Untuk mengekspor atau menyinkronkan artifact deployment, jalankan:

```bash
python pipeline_export_artifacts.py
```

Jika folder artifact berbeda, gunakan perintah berikut:

```bash
MODEL_ARTIFACT_DIR=./model_artifacts MODEL_ARTIFACT_OUT_DIR=./model_artifacts python pipeline_export_artifacts.py
```

Pipeline akan menulis ulang beberapa file artifact, seperti:

* `hybrid_twp90_model.joblib`
* `preprocessing_config.json`
* `test_predictions_hybrid.csv`
* `dashboard_twp90_history.csv`

Script pipeline ini tidak melakukan retraining otomatis, sehingga model final pada file joblib tetap tidak berubah.

---

## Kesimpulan

Dashboard Prediksi TWP90 ini merupakan bentuk implementasi dari hasil pemodelan time series berbasis **Hybrid SARIMAX + XGBoost Residual**. Dashboard dirancang agar hasil penelitian tidak hanya berhenti pada proses pemodelan, tetapi juga dapat digunakan secara interaktif untuk melakukan simulasi prediksi TWP90 pada periode tertentu.

Dengan adanya dashboard ini, pengguna dapat memilih periode target prediksi, mengisi variabel eksternal yang dibutuhkan, lalu melihat hasil prediksi TWP90 secara langsung melalui antarmuka Streamlit.
