# dashboardprediksi_TWP90

## Inti logika model

Metadata model menunjukkan skema:

- Model: Hybrid SARIMAX + XGBoost residual
- Skema evaluasi: walk-forward one-step-ahead forecasting
- Window: expanding
- Horizon: 1 bulan
- Target: `TWP90 (%)`
- Target XGBoost: residual SARIMAX log, bukan TWP90 langsung
- Input dashboard: variabel eksternal mentah dan TWP90 aktual untuk periode sebelum target
- Output dashboard: prediksi TWP90 bulan target beserta prediksi bulan-bulan sebelumnya yang dihitung berurutan

Dengan logika revisi ini, user tidak lagi hanya memilih “periode input”, tetapi memilih **periode TWP90 yang ingin diprediksi** melalui kalender. Sistem kemudian menghitung kebutuhan input sampai bulan sebelum target.

Contoh jika data historis terakhir adalah Desember 2025:

> Pilih target Februari 2026 -> user mengisi input Januari 2026 -> sistem menampilkan prediksi Januari dan Februari 2026.
>
> Pilih target Maret 2026 -> user mengisi input Januari dan Februari 2026 -> sistem menampilkan prediksi Januari, Februari, dan Maret 2026.
>
> Pilih target April 2026 -> user mengisi input Januari, Februari, dan Maret 2026 -> sistem menampilkan prediksi Januari, Februari, Maret, dan April 2026.

## Kenapa TWP90 aktual ikut diinput?

Pada model hybrid residual, XGBoost tidak memprediksi TWP90 secara langsung. XGBoost memprediksi residual SARIMAX pada skala log. Karena itu, ketika user ingin mencari TWP90 bulan Maret, nilai TWP90 aktual bulan Januari dan Februari dapat dipakai di balik layar untuk:

1. memperbarui state SARIMAX secara berurutan;
2. menghitung residual aktual SARIMAX bulan input;
3. memperbarui residual history XGBoost;
4. menjaga skema one-step-ahead agar tidak melompati bulan.

Dengan demikian, input TWP90 aktual bukan berarti model “membocorkan” nilai target bulan yang sedang dicari. TWP90 aktual hanya diminta untuk bulan-bulan sebelum periode target. Bulan target tetap diprediksi oleh model.

## File utama

1. `app.py`  
   Dashboard Streamlit. Desain tetap dipertahankan, tetapi logika menu prediksi diubah menjadi pilihan periode target TWP90. Dashboard membaca joblib/json dari `model_artifacts/` atau langsung dari folder yang sama dengan `app.py`.

2. `pipeline_export_artifacts.py`  
   Pipeline sinkronisasi artifact deployment. Script membaca `hybrid_twp90_model.joblib` dan `preprocessing_config.json`, lalu mengekspor metadata yang konsisten untuk dashboard. Script tidak melakukan retraining otomatis agar model final pada joblib tidak berubah.

3. `pipeline_export_artifacts.ipynb`  
   Versi notebook dari pipeline revisi.

4. `model_artifacts/`  
   Folder artifact yang disarankan. Minimal berisi:
   - `hybrid_twp90_model.joblib`
   - `preprocessing_config.json`
   - `raw_history.csv` untuk menjalankan menu prediksi

## Kenapa `raw_history.csv` tetap diperlukan?

Model pada artifact memakai differencing dan lag variabel eksternal, misalnya `BI-7Day-RR_lag1`, `Inflasi_lag1`, `IKK_lag6`, `Pertumbuhan Outstanding_lag12`, dan fitur residual SARIMAX. Karena itu, dashboard membutuhkan data historis mentah agar dapat membentuk fitur lag secara benar.

`raw_history.csv` harus berisi kolom tanggal, target, dan variabel eksternal historis berikut:

- `Month`
- `TWP90 (%)`
- `Outstanding Pinjaman (miliar RP)`
- `BI-7Day-RR`
- `Inflasi`
- `PDB (miliar Rp)`
- `Pertumbuhan Outstanding (YoY% atau MoM%)`
- `Indeks Keyakinan Konsumen (IKK)`
- `Nilai Tukar Rupiah terhadap USD`

Jika `raw_history.csv` belum tersedia, dashboard masih dapat membaca joblib untuk evaluasi dari `test_results_hybrid`, tetapi menu prediksi belum dapat menghitung fitur lag dan update residual secara lengkap.

## Cara menjalankan dashboard

```bash
pip install streamlit pandas numpy plotly joblib pmdarima xgboost scikit-learn statsmodels
streamlit run app.py
```

Struktur folder yang disarankan:

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

Dashboard juga dapat membaca `hybrid_twp90_model.joblib` dan `preprocessing_config.json` jika keduanya diletakkan satu folder dengan `app.py`.

## Cara menjalankan pipeline

```bash
python pipeline_export_artifacts.py
```

Atau jika folder artifact berbeda:

```bash
MODEL_ARTIFACT_DIR=./model_artifacts MODEL_ARTIFACT_OUT_DIR=./model_artifacts python pipeline_export_artifacts.py
```

Pipeline akan menulis ulang:

- `hybrid_twp90_model.joblib`
- `preprocessing_config.json`
- `test_predictions_hybrid.csv` jika tersedia dari joblib
- `dashboard_twp90_history.csv` jika data aktual/evaluasi tersedia

## Catatan input dashboard

Kolom persen diisi sebagai angka persen asli, bukan bentuk desimal internal model. Contoh:

- TWP90 4,32% ditulis `4.32`
- Inflasi 2,92% ditulis `2.92`
- BI Rate 5,75% ditulis `5.75`
- Pertumbuhan Outstanding 18,03% ditulis `18.03`

Nilai 0 dianggap belum diisi, sehingga prediksi tidak akan ditampilkan jika masih ada input 0 atau kosong.

Menu prediksi menyediakan pilihan periode target melalui kalender, bukan list/dropdown. Secara default rentang bulan target mengikuti `dashboard_max_selectable_months`. Jika ingin mengubah jumlah bulan target yang bisa dipilih, tambahkan nilai `dashboard_max_selectable_months` pada `preprocessing_config.json`.
