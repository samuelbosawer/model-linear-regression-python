# Model Linear Regression – Prediksi Kelembapan Tanah (Smart Garden AIoT)

Model *time-series forecasting* untuk memprediksi **kelembapan tanah 1, 3, 6, 12, dan 24 jam ke depan** dari data sensor Smart Garden AIoT. Tujuannya agar sistem bisa memberi **peringatan dini sebelum tanah benar-benar kering**, tidak hanya bereaksi setelah kelembapan turun di bawah batas seperti sistem *rule-based* sebelumnya (Tugas 2).

Model final adalah **Linear Regression** (scikit-learn). Koefisiennya disimpan ke JSON, sehingga dashboard Laravel bisa menghitung prediksi langsung di PHP tanpa library ML.

## Struktur Proyek

```
.
├── data/
│   └── dataset.csv               # data mentah ekspor dashboard (timestamp UTC)
├── ml/
│   ├── prepare.py                # preprocessing, fitur, dan target
│   ├── train.py                  # perbandingan model + pelatihan model final
│   └── evaluate_decision.py      # evaluasi keputusan AI vs rule-based
├── results/                      # tabel hasil, log, dan model final (.json)
└── fig/                          # grafik untuk laporan (dibuat otomatis)
```

## Dataset

`data/dataset.csv` berisi ±130 ribu baris log sensor dengan kolom:

| Kolom | Keterangan |
|---|---|
| `timestamp` | waktu pencatatan (UTC) |
| `soil_percent` | kelembapan tanah (%) |
| `soil_raw` | nilai ADC sensor kelembapan (12-bit, 0–4095) |
| `soil_temperature` | suhu tanah (°C) |
| `is_water_empty` | status tangki air kosong |
| `pump_status`, `pump_command` | status/perintah pompa |
| `status`, `alarm` | status sistem (mis. `LEMBAB AMAN`, `MULAI AUTO`) |

## Alur Pengolahan Data (`ml/prepare.py`)

1. **Load** – baca CSV, ubah timestamp ke WIB (Asia/Jakarta), urutkan, buang duplikat.
2. **Clean** – buang data sebelum sensor dikalibrasi (`soil_raw` < 500), tandai periode uji fungsional sensor (28–29 Okt 2026).
3. **Resample** – rata-rata per 10 menit; celah ≤ 30 menit diinterpolasi.
4. **Fitur** – kelembapan sekarang dan perubahan kelembapan 1, 3, 6, 12, 24 jam terakhir:
   `soil_now`, `soil_diff_1h`, `soil_diff_3h`, `soil_diff_6h`, `soil_diff_12h`, `soil_diff_24h`.
5. **Target** – *perubahan* kelembapan H jam ke depan. Prediksi akhir = `soil_now + perubahan` (dibatasi 0–100%).
   Sampel yang jendelanya terkena penyiraman, data hilang, atau periode uji sensor dibuang.

## Metode Evaluasi

Evaluasi menggunakan **walk-forward harian** pada periode uji **4–12 Nov 2026 (9 hari)**: setiap hari model dilatih ulang hanya dengan data sebelum hari tersebut, lalu memprediksi hari itu (meniru *retraining* setiap malam).

Model yang dibandingkan:

- Persistence (baseline – menganggap kelembapan tidak berubah)
- **Linear Regression**
- Linear Regression + suhu & jam
- Random Forest
- Gradient Boosting (HistGradientBoosting)
- MLP Neural Network

## Hasil

### MAE (% kelembapan) per horizon – makin kecil makin baik

| Model | 1 jam | 3 jam | 6 jam | 12 jam | 24 jam | Rata-rata | vs baseline |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | 0.314 | 0.516 | 0.673 | 0.761 | **0.932** | **0.639** | **+37.2%** |
| Linear Regression + suhu & jam | 0.321 | **0.515** | 0.686 | 0.786 | 0.967 | 0.655 | +35.7% |
| MLP Neural Network | 0.399 | 0.612 | 0.897 | 1.015 | 1.135 | 0.812 | +20.3% |
| Random Forest | 0.384 | 0.664 | 0.886 | 1.236 | 1.065 | 0.847 | +16.8% |
| Gradient Boosting | 0.445 | 0.831 | 1.085 | **0.715** | 1.262 | 0.868 | +14.8% |
| Persistence (baseline) | **0.311** | 0.558 | 0.848 | 1.171 | 2.203 | 1.018 | – |

Linear Regression dipilih karena memiliki **rata-rata MAE terendah** di semua horizon, sederhana, dan mudah diterapkan di dashboard. Penambahan fitur suhu & jam tidak memperbaiki akurasi.

### Horizon utama 12 jam (Linear Regression)

| Metrik | Linear Regression | Baseline |
|---|---|---|
| MAE | 0.761 | 1.171 |
| RMSE | 0.945 | 1.443 |
| R² | 0.869 | 0.696 |

### Keputusan AI vs Rule-based (`ml/evaluate_decision.py`)

- **Rule-based (Tugas 2):** butuh air jika kelembapan sekarang < 55%.
- **AI hybrid:** peringatan jika peluang kelembapan 12 jam lagi < 55% ≥ 0,70 (dari prediksi Linear Regression dan sigma galat walk-forward), ditambah aturan lama sebagai pengaman.

| Sistem | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-based (Tugas 2) | 0.930 | 1.000 | 0.639 | 0.780 |
| **AI hybrid (Final Project)** | **0.947** | **1.000** | **0.728** | **0.842** |

AI hybrid memberi peringatan pertama **±2,2 jam lebih awal** daripada aturan lama (05 Nov 03:20 vs 05:30 WIB), tanpa alarm palsu pada 7–12 Nov.

## Cara Menjalankan

### Prasyarat

- Python 3.10+ (diuji dengan Python 3.14)
- Library: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Langkah

```bash
# 1. Latih & bandingkan model, simpan model final dan grafik
python ml/train.py            # grafik ditampilkan di jendela
python ml/train.py --no-show  # grafik hanya disimpan ke fig/

# 2. Evaluasi keputusan AI vs rule-based (butuh results/soil_forecast_model.json dari langkah 1)
python ml/evaluate_decision.py
```

## Output

| File | Isi |
|---|---|
| `results/dataset_10min.csv` | data hasil resample 10 menit |
| `results/model_comparison.csv` | MAE, RMSE, R² semua model per horizon |
| `results/model_summary.csv` | rata-rata MAE dan peringkat model |
| `results/metrics_12h.csv` | metrik horizon 12 jam |
| `results/walkforward_predictions.csv` | prediksi vs aktual (12 jam) |
| `results/soil_forecast_model.json` | **model final** (intercept, koefisien, sigma per horizon) untuk dashboard Laravel |
| `results/decision_comparison.csv` | metrik klasifikasi AI vs rule-based |
| `results/decision_replay.csv` | keputusan per titik waktu pada periode uji |
| `results/decision_summary.json` | ringkasan lead time dan alarm |
| `fig/fig_dataset.png` | grafik kelembapan & suhu tanah |
| `fig/fig_mae_horizon.png` | perbandingan MAE per horizon |
| `fig/fig_train_pred_12h.png` | prediksi 12 jam vs aktual |
| `fig/fig_decision_timeline.png` | timeline peringatan AI vs rule-based |
| `fig/fig_pred_vs_actual.png` | prediksi vs aktual seluruh periode uji |
| `fig/fig_confusion.png` | confusion matrix kedua sistem |

## Pemakaian Model di Dashboard

Untuk setiap horizon `H`, prediksi dihitung dari koefisien di `soil_forecast_model.json`:

```
perubahan_H = intercept + Σ (coef[fitur] × nilai_fitur)
prediksi_H  = clip(soil_now + perubahan_H, 0, 100)
rentang 95% = prediksi_H ± 1,96 × sigma
```

Tanah dianggap **akan kering** jika prediksi berada di bawah `dry_threshold` (55%).
