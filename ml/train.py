"""
Pelatihan dan evaluasi model prediksi kelembapan tanah (time-series forecasting).

Evaluasi memakai walk-forward: setiap hari pada periode uji (4-12 Nov 2026), model dilatih ulang
hanya dengan data sebelum hari itu, lalu dipakai untuk memprediksi hari tersebut.
Cara ini meniru pemakaian nyata, ketika model dilatih ulang setiap malam.

Pemakaian:
    python ml/train.py            -> latih, cetak penjelasan rinci, simpan & tampilkan grafik
    python ml/train.py --no-show  -> sama, tetapi grafik hanya disimpan (tidak dibuka di jendela)

Output:
- results/model_comparison.csv       : MAE semua model untuk setiap horizon
- results/metrics_12h.csv            : MAE, RMSE, R2 untuk horizon 12 jam
- results/walkforward_predictions.csv: prediksi vs aktual (horizon 12 jam)
- results/soil_forecast_model.json   : model final (koefisien) yang dipakai dashboard Laravel
- fig/*.png                          : grafik untuk laporan

Alur program (fungsi main):
    Tahap 1  Muat & bersihkan data sensor, resample ke 10 menit.
    Tahap 2  Bandingkan 6 model di 5 horizon dengan evaluasi walk-forward.
    Tahap 3  Ringkas hasil (tabel MAE, peringkat, fokus horizon 12 jam).
    Tahap 4  Latih model final Linear Regression dengan semua data, simpan ke JSON.
    Tahap 5  Buat grafik, simpan ke fig/, lalu tampilkan di layar.
"""
# ============================================================================
# BAGIAN A. IMPORT LIBRARY
# ============================================================================
import argparse   # membaca argumen dari terminal, misalnya --no-show
import json       # menyimpan model final ke file .json
import sys        # mengatur path import dan encoding output terminal
import time       # mengukur lama proses tiap model
import warnings   # menyembunyikan peringatan yang tidak penting dari library
from pathlib import Path  # menulis path file yang aman untuk Windows/Linux

import matplotlib

# Argumen terminal dibaca SEBELUM pyplot di-import, karena backend grafik harus dipilih lebih dulu.
# Backend "Agg" = hanya menyimpan gambar ke file, tanpa membuka jendela.
ARGS = argparse.ArgumentParser(description="Latih model prediksi kelembapan tanah")
ARGS.add_argument("--no-show", action="store_true", help="jangan buka jendela grafik, hanya simpan PNG")
ARGS = ARGS.parse_args()
if ARGS.no_show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  membuat grafik
import numpy as np  # noqa: E402               operasi numerik (array, akar, clip)
import pandas as pd  # noqa: E402              tabel data deret waktu
# Model-model yang dibandingkan:
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
# Metrik evaluasi dan alat bantu:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Fungsi preprocessing ada di ml/prepare.py (folder yang sama dengan file ini).
sys.path.insert(0, str(Path(__file__).parent))
from prepare import DRY_THRESHOLD, FEATURES, FEATURES_WIBH_TEMP, clean, load_raw, make_dataset, make_features, resample  # noqa: E402

# Pastikan karakter non-ASCII tercetak dengan benar di terminal Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================================
# BAGIAN B. KONFIGURASI
# ============================================================================
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]          # folder utama proyek
RESULTS, FIG = ROOT / "results", ROOT / "fig"       # folder output tabel dan grafik
RESULTS.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

TEST_DAYS = pd.date_range("2026-11-04", "2026-11-12", freq="D")  # 9 hari periode uji
HORIZONS = [1, 3, 6, 12, 24]   # berapa jam ke depan yang diprediksi
MAIN_H = 12                    # horizon utama yang dipakai dashboard untuk peringatan dini

# Kandidat model. Ditulis sebagai lambda supaya setiap pelatihan ulang memakai model baru (bukan sisa
# latihan sebelumnya).
CANDIDATES = {
    # Regresi linear biasa: prediksi = intercept + jumlah (koefisien x fitur).
    "Linear Regression": lambda: LinearRegression(),
    # Kumpulan 300 pohon keputusan; hasil akhirnya rata-rata semua pohon.
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=42),
    # Pohon keputusan yang dibangun bertahap, tiap pohon memperbaiki kesalahan pohon sebelumnya.
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=42),
    # Jaringan saraf 2 lapisan tersembunyi (32 dan 16 neuron). Fitur dinormalisasi dulu dengan StandardScaler.
    "MLP Neural Network": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, early_stopping=True, random_state=42)),
}

# Penjelasan singkat tiap fitur, hanya untuk dicetak di terminal.
FEATURE_DESC = {
    "soil_now": "kelembapan tanah saat ini (%)",
    "soil_diff_1h": "perubahan kelembapan 1 jam terakhir",
    "soil_diff_3h": "perubahan kelembapan 3 jam terakhir",
    "soil_diff_6h": "perubahan kelembapan 6 jam terakhir",
    "soil_diff_12h": "perubahan kelembapan 12 jam terakhir",
    "soil_diff_24h": "perubahan kelembapan 24 jam terakhir",
}

# ============================================================================
# BAGIAN C. FUNGSI BANTU TAMPILAN TERMINAL
# ============================================================================
W = 78  # lebar garis pemisah di terminal


def banner(title: str):
    """Judul tahap besar, diapit garis '='."""
    print("\n" + "=" * W)
    print(f" {title}")
    print("=" * W)


def sub(title: str):
    """Sub-judul di dalam satu tahap, diikuti garis '-'."""
    print("\n" + f"--- {title} ".ljust(W, "-"))


def info(text: str, indent: int = 2):
    """Cetak satu baris dengan indentasi."""
    print(" " * indent + text)


def bar(value: float, vmax: float, width: int = 30) -> str:
    """Diagram batang sederhana dari karakter '#', panjangnya sebanding dengan value/vmax."""
    n = 0 if vmax <= 0 else int(round(width * value / vmax))
    return "#" * n + "." * (width - n)


# ============================================================================
# BAGIAN D. FUNGSI INTI EVALUASI
# ============================================================================
def walk_forward(ds: pd.DataFrame, horizon: int, make_model, cols) -> pd.DataFrame:
    """Evaluasi walk-forward: latih ulang model setiap hari uji, lalu prediksi hari itu.

    ds         : dataset hasil make_dataset() (fitur + target)
    horizon    : horizon prediksi dalam jam
    make_model : fungsi pembuat model; None berarti baseline persistence
    cols       : daftar nama fitur yang dipakai
    Return     : tabel berisi soil_now, actual (nilai sebenarnya), dan pred (prediksi).
    """
    parts = []
    for day in TEST_DAYS:
        # Data latih: hanya sampel yang targetnya sudah diketahui sebelum hari `day` dimulai
        # (sampel berumur >= horizon jam). Ini mencegah model "mengintip" masa depan.
        train = ds[ds.index < day - pd.Timedelta(hours=horizon)]
        # Data uji: semua sampel pada hari `day`.
        test = ds[(ds.index >= day) & (ds.index < day + pd.Timedelta(days=1))]
        if test.empty:
            continue
        if make_model is None:
            # Baseline persistence: menebak tidak ada perubahan (delta = 0).
            delta = np.zeros(len(test))
        else:
            model = make_model()
            model.fit(train[cols], train["target_delta"])  # model belajar memprediksi PERUBAHAN
            delta = model.predict(test[cols])
        parts.append(pd.DataFrame({
            "soil_now": test["soil_now"],
            "actual": test["target_soil"],
            # Prediksi akhir = kelembapan sekarang + perubahan, dibatasi 0-100%.
            "pred": (test["soil_now"] + delta).clip(0, 100),
        }, index=test.index))
    return pd.concat(parts)


def scores(p: pd.DataFrame) -> dict:
    """Hitung metrik akurasi dari tabel prediksi.

    MAE  : rata-rata |aktual - prediksi|, satuan % kelembapan (makin kecil makin baik).
    RMSE : akar rata-rata kuadrat galat; lebih menghukum galat besar.
    R2   : proporsi variasi data yang dijelaskan model (1 = sempurna, 0 = setara rata-rata, <0 = buruk).
    n    : jumlah sampel uji.
    """
    return {
        "MAE": mean_absolute_error(p["actual"], p["pred"]),
        "RMSE": float(np.sqrt(mean_squared_error(p["actual"], p["pred"]))),
        "R2": r2_score(p["actual"], p["pred"]),
        "n": len(p),
    }


# ============================================================================
# BAGIAN E. PROGRAM UTAMA
# ============================================================================
def main():
    t0 = time.time()  # waktu mulai, untuk menghitung total durasi
    banner("SMART GARDEN AIoT - PELATIHAN MODEL PREDIKSI KELEMBAPAN TANAH")
    info("Tujuan : memprediksi kelembapan tanah 1, 3, 6, 12, dan 24 jam ke depan,")
    info("         supaya sistem bisa memberi peringatan SEBELUM tanah benar-benar kering.")
    info(f"Batas kering yang dipakai: {DRY_THRESHOLD}%")

    # ------------------------------------------------------------------
    # TAHAP 1. Muat dan bersihkan data
    # ------------------------------------------------------------------
    banner("TAHAP 1/5 - MEMUAT & MEMBERSIHKAN DATA")
    # load_raw: baca CSV, ubah timestamp UTC ke WIB, urutkan, buang duplikat.
    raw0 = load_raw(ROOT / "data" / "dataset.csv")
    info(f"File sumber          : data/dataset.csv")
    info(f"Jumlah baris mentah  : {len(raw0):,}")
    info(f"Rentang waktu (WIB)  : {raw0.index.min():%d %b %Y %H:%M}  s.d.  {raw0.index.max():%d %b %Y %H:%M}")
    # clean: buang data sebelum sensor dikalibrasi dan tandai periode uji sensor.
    raw = clean(raw0)
    info(f"Setelah dibersihkan  : {len(raw):,} baris  ({len(raw0) - len(raw):,} baris dibuang)")
    info("  -> Data sebelum sensor dikalibrasi (soil_raw masih 0-100, bukan ADC 12-bit) dibuang.")
    info(f"  -> {int(raw['in_test_window'].sum()):,} baris ditandai sebagai periode uji sensor (tidak dipakai untuk target).")

    # resample: rata-rata per 10 menit agar interval data seragam; celah pendek diinterpolasi.
    r = resample(raw)
    r.to_csv(RESULTS / "dataset_10min.csv", index_label="timestamp_wit")
    sub("Resample ke interval 10 menit")
    info(f"Jumlah titik 10 menit: {len(r):,}")
    info(f"Titik hilang (gap)   : {int(r['missing'].sum()):,}  (celah <= 30 menit sudah diinterpolasi)")
    info(f"Kejadian pompa ON    : {int(r['pump_on'].sum()):,} titik")
    info(f"Kelembapan tanah     : min {r['soil'].min():.1f}%  | rata-rata {r['soil'].mean():.1f}%  | maks {r['soil'].max():.1f}%")
    info(f"Suhu tanah           : min {r['temp'].min():.1f}C  | rata-rata {r['temp'].mean():.1f}C  | maks {r['temp'].max():.1f}C")
    below = f"Waktu di bawah {DRY_THRESHOLD}%"
    info(f"{below:<21}: {(r['soil'] < DRY_THRESHOLD).mean() * 100:.1f}% dari seluruh waktu")

    sub("Fitur yang dipakai model")
    for f in FEATURES:
        info(f"- {f:<15} : {FEATURE_DESC.get(f, '')}")
    info("Fitur tambahan yang diuji (varian '+ suhu & jam'): temp_now, temp_diff_1h, hour_sin, hour_cos")
    info("Target model = PERUBAHAN kelembapan H jam ke depan (bukan nilai absolut),")
    info("lalu prediksi akhir = kelembapan sekarang + perubahan yang diprediksi.")

    # ------------------------------------------------------------------
    # TAHAP 2. Bandingkan semua model di semua horizon
    # ------------------------------------------------------------------
    banner("TAHAP 2/5 - PERBANDINGAN MODEL (EVALUASI WALK-FORWARD)")
    info(f"Periode uji : {TEST_DAYS[0]:%d %b} - {TEST_DAYS[-1]:%d %b %Y} ({len(TEST_DAYS)} hari)")
    info("Cara kerja  : untuk setiap hari uji, model dilatih ulang HANYA dengan data sebelum")
    info("              hari itu, lalu memprediksi hari tersebut (meniru retraining tiap malam).")
    info("Baseline    : 'Persistence' = menganggap kelembapan H jam lagi sama dengan sekarang.")
    info("              Model AI dianggap berguna jika MAE-nya lebih kecil dari baseline.")

    # rows       : semua hasil metrik (untuk tabel perbandingan)
    # main_preds : prediksi tiap model pada horizon utama (12 jam), untuk grafik
    # lr_wf      : prediksi walk-forward Linear Regression per horizon, untuk menghitung sigma di Tahap 4
    rows, main_preds, lr_wf = [], {}, {}
    for h in HORIZONS:
        # Buat dataset fitur+target untuk horizon h; sampel yang terganggu penyiraman/gap dibuang.
        ds = make_dataset(r, h)
        sub(f"Horizon {h} jam  |  sampel valid: {len(ds):,}")
        # Daftar (nama, pembuat model, fitur) yang akan diuji.
        entries = [("Persistence (baseline)", None, FEATURES)]
        entries += [(name, fn, FEATURES) for name, fn in CANDIDATES.items()]
        entries += [("Linear Regression + suhu & jam", CANDIDATES["Linear Regression"], FEATURES_WIBH_TEMP)]
        h_rows = []
        for name, fn, cols in entries:
            ts = time.time()
            p = walk_forward(ds, h, fn, cols)   # jalankan evaluasi walk-forward
            sc = scores(p)                      # hitung MAE, RMSE, R2
            row = {"horizon_h": h, "model": name, **sc}
            rows.append(row); h_rows.append(row)
            if h == MAIN_H:
                main_preds[name] = p
            if name == "Linear Regression":
                lr_wf[h] = p
            info(f"{name:<32} MAE {sc['MAE']:6.3f}  RMSE {sc['RMSE']:6.3f}  R2 {sc['R2']:6.3f}  ({time.time() - ts:4.1f} dtk)")
        # Bandingkan model terbaik di horizon ini dengan baseline.
        base_mae = h_rows[0]["MAE"]
        best_row = min(h_rows, key=lambda x: x["MAE"])
        gain = (1 - best_row["MAE"] / base_mae) * 100
        info(f"=> Terbaik: {best_row['model']} (MAE {best_row['MAE']:.3f}%, "
             f"{'lebih baik' if gain >= 0 else 'lebih buruk'} {abs(gain):.1f}% dari baseline)")

    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # TAHAP 3. Ringkasan hasil
    # ------------------------------------------------------------------
    # piv_all: tabel MAE dengan baris = model, kolom = horizon.
    piv_all = comp.pivot(index="model", columns="horizon_h", values="MAE")
    # summary: rata-rata MAE semua horizon, jumlah "menang", dan persen perbaikan terhadap baseline.
    summary = pd.DataFrame({
        "mean_MAE": piv_all.mean(axis=1),
        "wins": piv_all.eq(piv_all.min(axis=0), axis=1).sum(axis=1),
        "improvement_vs_baseline_%": (1 - piv_all.mean(axis=1) / piv_all.loc["Persistence (baseline)"].mean()) * 100,
    }).sort_values("mean_MAE")
    summary.to_csv(RESULTS / "model_summary.csv")

    banner("TAHAP 3/5 - RINGKASAN HASIL")
    sub("Tabel MAE (% kelembapan) per horizon - makin kecil makin baik")
    print(piv_all.loc[summary.index].round(3).rename(columns=lambda c: f"{c} jam").to_string())
    sub("Peringkat model (rata-rata MAE semua horizon)")
    vmax = summary["mean_MAE"].max()
    for i, (name, s) in enumerate(summary.iterrows(), 1):
        info(f"{i}. {name:<32} {bar(s['mean_MAE'], vmax, 24)} {s['mean_MAE']:.3f}  "
             f"menang {int(s['wins'])}x  vs baseline {s['improvement_vs_baseline_%']:+.1f}%")
    info("Keterangan:")
    info("  MAE  = rata-rata selisih absolut prediksi vs aktual, dalam satuan % kelembapan.", 4)
    info("  RMSE = seperti MAE tetapi lebih menghukum galat besar.", 4)
    info("  R2   = seberapa banyak variasi data yang dijelaskan model (1 = sempurna, <0 = buruk).", 4)
    info("  menang = jumlah horizon di mana model tersebut memiliki MAE terkecil.", 4)

    # Metrik khusus horizon 12 jam (dipakai di laporan dan JSON model).
    m12 = comp[comp.horizon_h == MAIN_H].set_index("model")[["MAE", "RMSE", "R2", "n"]]
    m12.to_csv(RESULTS / "metrics_12h.csv")

    # Simpan prediksi 12 jam Linear Regression + baseline untuk dianalisis/diplot.
    best = main_preds["Linear Regression"].copy()
    best["baseline"] = main_preds["Persistence (baseline)"]["pred"]
    best.to_csv(RESULTS / "walkforward_predictions.csv", index_label="timestamp_wit")

    lr12, base12 = m12.loc["Linear Regression"], m12.loc["Persistence (baseline)"]
    sub(f"Fokus horizon utama {MAIN_H} jam (dipakai untuk peringatan dini di dashboard)")
    info(f"Linear Regression : MAE {lr12['MAE']:.3f}%  RMSE {lr12['RMSE']:.3f}  R2 {lr12['R2']:.3f}  (n uji = {int(lr12['n'])})")
    info(f"Baseline          : MAE {base12['MAE']:.3f}%  RMSE {base12['RMSE']:.3f}  R2 {base12['R2']:.3f}")
    info(f"Artinya: rata-rata prediksi 12 jam ke depan meleset sekitar {lr12['MAE']:.2f} poin persen,")
    info(f"         {(1 - lr12['MAE'] / base12['MAE']) * 100:.1f}% lebih akurat daripada sekadar menebak 'tidak berubah'.")
    err = best["actual"] - best["pred"]  # galat tiap prediksi
    info(f"         {(err.abs() <= 1).mean() * 100:.1f}% prediksi meleset <= 1 poin, "
         f"{(err.abs() <= 2).mean() * 100:.1f}% meleset <= 2 poin; galat terbesar {err.abs().max():.2f} poin.")
    info("Alasan memilih Linear Regression: rata-rata MAE terendah di semua horizon, sederhana,")
    info("dan koefisiennya mudah dihitung ulang langsung di PHP/Laravel tanpa library ML.")

    # ------------------------------------------------------------------
    # TAHAP 4. Latih model final dengan SEMUA data, simpan ke JSON
    # ------------------------------------------------------------------
    banner("TAHAP 4/5 - MELATIH MODEL FINAL (SEMUA DATA) UNTUK DASHBOARD")
    # Struktur JSON ini dibaca oleh dashboard Laravel (perintah ml:predict).
    model_json = {
        "name": "soil_moisture_forecast",
        "algorithm": "LinearRegression (scikit-learn), target = perubahan kelembapan H jam ke depan",
        "trained_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(timespec="seconds"),
        "data_until": str(r.index.max()),
        "resample": "10min",
        "features": FEATURES,
        "dry_threshold": DRY_THRESHOLD,
        "horizons": {},
    }
    for h in HORIZONS:
        ds = make_dataset(r, h)
        lr = LinearRegression().fit(ds[FEATURES], ds["target_delta"])
        # Sigma = simpangan baku galat walk-forward (data yang belum pernah dilihat model), bukan galat
        # latih. Dipakai dashboard untuk menampilkan rentang ketidakpastian prediksi.
        wf = lr_wf[h]
        sigma = float(np.std(wf["actual"] - wf["pred"], ddof=1))
        model_json["horizons"][str(h)] = {
            "intercept": float(lr.intercept_),
            "coef": dict(zip(FEATURES, map(float, lr.coef_))),
            "sigma": sigma,
            "n_train": len(ds),
        }
        info(f"Horizon {h:>2} jam : dilatih dengan {len(ds):>5,} sampel | sigma (ketidakpastian) = +/-{sigma:.3f}%")

    model_json["metrics_12h"] = {
        "evaluation": "walk-forward harian 4-12 Nov 2026",
        "mae": round(float(lr12["MAE"]), 3), "rmse": round(float(lr12["RMSE"]), 3), "r2": round(float(lr12["R2"]), 3),
        "baseline_mae": round(float(base12["MAE"]), 3), "n_test": int(lr12["n"]),
    }
    (RESULTS / "soil_forecast_model.json").write_text(json.dumps(model_json, indent=2))

    # Cetak persamaan model 12 jam agar mudah dijelaskan di laporan.
    m = model_json["horizons"][str(MAIN_H)]
    sub(f"Persamaan model horizon {MAIN_H} jam")
    info(f"perubahan_{MAIN_H}h = {m['intercept']:+.4f}")
    for f, c in m["coef"].items():
        info(f"               {c:+.4f} x {f}")
    info(f"prediksi_{MAIN_H}h  = soil_now + perubahan_{MAIN_H}h   (dibatasi 0-100%)")
    sub("Cara membaca koefisien")
    # Diurutkan dari koefisien dengan pengaruh terbesar (nilai absolut).
    for f, c in sorted(m["coef"].items(), key=lambda kv: -abs(kv[1])):
        arah = "menaikkan" if c > 0 else "menurunkan"
        info(f"- {f:<14}: setiap +1 poin {arah} prediksi perubahan sebesar {abs(c):.3f} poin")
    info(f"Rentang keyakinan ~95% prediksi {MAIN_H} jam: +/-{1.96 * m['sigma']:.2f} poin persen (1,96 x sigma).")

    # Contoh pemakaian model: hitung prediksi manual dari baris data terakhir yang fiturnya lengkap,
    # persis seperti yang dilakukan dashboard dengan koefisien JSON.
    feats = make_features(r).dropna(subset=FEATURES)
    if not feats.empty:
        last = feats.iloc[-1]
        sub(f"Contoh prediksi dari data terakhir ({feats.index[-1]:%d %b %Y %H:%M} WIB)")
        info(f"Kelembapan sekarang: {last['soil_now']:.1f}%")
        for h in HORIZONS:
            hm = model_json["horizons"][str(h)]
            delta = hm["intercept"] + sum(hm["coef"][f] * last[f] for f in FEATURES)
            pred = float(np.clip(last["soil_now"] + delta, 0, 100))
            status = "KERING" if pred < DRY_THRESHOLD else "aman"
            info(f"  {h:>2} jam lagi -> {pred:5.1f}%  (+/-{1.96 * hm['sigma']:.1f})  [{status}]")

    # ------------------------------------------------------------------
    # TAHAP 5. Grafik
    # ------------------------------------------------------------------
    banner("TAHAP 5/5 - MEMBUAT GRAFIK")
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    figs = []  # daftar (nama file, penjelasan) untuk dicetak di akhir

    # 5a. Grafik dataset: kelembapan (atas) dan suhu (bawah), titik merah = pompa menyala.
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax[0].plot(r.index, r["soil"], lw=0.8, color="#2b6cb0", label="Kelembapan tanah (%)")
    pumps = r.index[r["pump_on"] > 0]
    ax[0].scatter(pumps, [104] * len(pumps), s=6, color="#c53030", label="Pompa menyala")
    ax[0].axhline(DRY_THRESHOLD, ls="--", lw=0.8, color="gray", label=f"Batas kering {DRY_THRESHOLD}%")
    ax[0].set_ylabel("Kelembapan (%)"); ax[0].set_ylim(15, 108); ax[0].legend(loc="lower left", fontsize=8)
    ax[1].plot(r.index, r["temp"], lw=0.8, color="#dd6b20")
    ax[1].set_ylabel("Suhu tanah (°C)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Data sensor setelah resample 10 menit (25 Okt – 12 Nov 2026, WIB)")
    fig.tight_layout(); fig.savefig(FIG / "fig_dataset.png", dpi=170)
    figs.append(("fig_dataset.png", "Kelembapan & suhu tanah per 10 menit, titik merah = pompa menyala."))

    # 5b. Diagram batang MAE: kelompok per horizon, satu batang per model.
    piv = comp.pivot(index="horizon_h", columns="model", values="MAE")
    order = ["Persistence (baseline)", "Linear Regression", "Random Forest", "Gradient Boosting", "MLP Neural Network"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.16
    x = np.arange(len(piv.index))
    colors = ["#a0aec0", "#2f855a", "#3182ce", "#d69e2e", "#805ad5"]
    for i, m_ in enumerate(order):
        # Geser posisi tiap batang agar 5 model berdampingan dalam satu kelompok horizon.
        ax.bar(x + (i - 2) * width, piv[m_], width, label=m_, color=colors[i])
    ax.set_xticks(x, [f"{h} jam" for h in piv.index])
    ax.set_ylabel("MAE (% kelembapan)"); ax.set_xlabel("Horizon prediksi")
    ax.set_title("Perbandingan MAE model (evaluasi walk-forward, 4–12 Nov 2026)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_mae_horizon.png", dpi=170)
    figs.append(("fig_mae_horizon.png", "MAE tiap model per horizon; batang lebih pendek = lebih akurat."))

    # 5c. Prediksi 12 jam vs aktual. Kiri: deret waktu (titik prediksi digeser +12 jam ke waktu target,
    # sehingga bisa langsung dibandingkan dengan garis aktual). Kanan: scatter aktual vs prediksi.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(best.index + pd.Timedelta(hours=MAIN_H), best["actual"], color="#2d3748", lw=1, label="Aktual")
    ax[0].plot(best.index + pd.Timedelta(hours=MAIN_H), best["pred"], ".", ms=2, color="#2f855a", label="Prediksi Linear Regression")
    ax[0].plot(best.index + pd.Timedelta(hours=MAIN_H), best["baseline"], ".", ms=1.5, color="#a0aec0", label="Baseline (persistence)")
    ax[0].set_ylabel("Kelembapan (%)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, markerscale=4)
    ax[0].set_title(f"Prediksi {MAIN_H} jam (digeser ke waktu target)")
    lo = min(best["actual"].min(), best["pred"].min()) - 1
    hi = max(best["actual"].max(), best["pred"].max()) + 1
    ax[1].scatter(best["actual"], best["baseline"], s=4, color="#a0aec0", label=f"Baseline MAE {base12['MAE']:.2f}")
    ax[1].scatter(best["actual"], best["pred"], s=4, color="#2f855a", label=f"LR MAE {lr12['MAE']:.2f}")
    ax[1].plot([lo, hi], [lo, hi], color="black", lw=0.8)  # garis diagonal = prediksi sempurna
    ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel("Aktual (%)"); ax[1].set_ylabel("Prediksi (%)"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, markerscale=3); ax[1].set_title("Aktual vs prediksi")
    fig.tight_layout(); fig.savefig(FIG / "fig_train_pred_12h.png", dpi=170)
    figs.append(("fig_train_pred_12h.png", "Prediksi 12 jam vs aktual; titik dekat garis diagonal = prediksi tepat."))

    for name, desc in figs:
        info(f"[OK] fig/{name:<24} {desc}")

    banner("SELESAI")
    info("File hasil:")
    for f in ["model_comparison.csv", "model_summary.csv", "metrics_12h.csv",
              "walkforward_predictions.csv", "dataset_10min.csv", "soil_forecast_model.json"]:
        info(f"- results/{f}")
    info(f"Total waktu: {time.time() - t0:.1f} detik")
    info("Langkah berikutnya: python ml/evaluate_decision.py  (bandingkan keputusan AI vs rule-based)")

    # Tampilkan semua grafik di jendela (kecuali --no-show). plt.show() menahan program
    # sampai semua jendela grafik ditutup.
    if ARGS.no_show or matplotlib.get_backend().lower() == "agg":
        plt.close("all")
    else:
        info("\nMenampilkan grafik... tutup semua jendela grafik untuk mengakhiri program.")
        plt.show()


# Jalankan main() hanya jika file ini dieksekusi langsung (bukan di-import).
if __name__ == "__main__":
    main()
