"""
Preprocessing dataset Smart Garden AIoT.

Langkah:
1. Baca CSV ekspor dashboard, ubah timestamp ke zona WIB (Asia/Jakarta).
2. Buang data sebelum kalibrasi sensor (soil_raw masih bernilai 0-100).
3. Resample ke interval 10 menit (rata-rata), supaya interval data seragam.
4. Buat fitur deret waktu: lag, rata-rata bergerak, kemiringan (slope), suhu, jam, riwayat pompa.
5. Buat target: perubahan kelembapan tanah H jam ke depan.
   Sampel yang jendela targetnya terkena penyiraman, data hilang (gap), atau periode uji sensor dibuang.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- konstanta
STEP = "10min"        # interval resample
STEPS_PER_HOUR = 6    # 60 menit / 10 menit = 6 titik per jam
DRY_THRESHOLD = 55    # kelembapan < 55% dianggap kering (sama dengan aturan firmware lama)

# Periode uji fungsional Tugas 2 (sensor dicabut/dipindah dengan sengaja), tidak mewakili kondisi alami.
TEST_WINDOWS = [("2026-10-28 10:00", "2026-10-29 04:00")]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Langkah 1: baca CSV mentah dan siapkan kolom dasar."""
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig membuang karakter BOM di awal file
    # Timestamp di CSV dalam UTC ("Z"); ubah ke WIB lalu hapus info zona agar mudah diolah.
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    # Urutkan berdasarkan waktu, buang baris dengan waktu kembar, jadikan waktu sebagai index.
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    # Ubah status teks menjadi angka 0/1 supaya bisa dihitung.
    df["pump_on"] = (df["pump_status"] == "ON").astype(int)
    df["water_empty"] = df["is_water_empty"].astype(str).str.lower().eq("true").astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Langkah 2: buang data sebelum kalibrasi dan tandai periode uji sensor."""
    # Sebelum kalibrasi, soil_raw tercatat 0-100 (bukan nilai ADC 12 bit 0-4095).
    # Data dianggap valid mulai baris pertama yang soil_raw-nya >= 500.
    calibrated_from = df.index[df["soil_raw"] >= 500].min()
    df = df[df.index >= calibrated_from].copy()
    # Kolom penanda: 1 jika baris berada di periode uji sensor.
    df["in_test_window"] = 0
    for start, end in TEST_WINDOWS:
        df.loc[start:end, "in_test_window"] = 1
    return df


def resample(df: pd.DataFrame) -> pd.DataFrame:
    """Langkah 3: ringkas data menjadi satu baris per 10 menit."""
    r = pd.DataFrame({
        "soil": df["soil_percent"].resample(STEP).mean(),          # rata-rata kelembapan
        "temp": df["soil_temperature"].resample(STEP).mean(),      # rata-rata suhu
        "pump_on": df["pump_on"].resample(STEP).max(),             # 1 jika pompa sempat menyala
        "water_empty": df["water_empty"].resample(STEP).max(),     # 1 jika tangki sempat kosong
        "n": df["soil_percent"].resample(STEP).size(),             # jumlah data mentah di slot ini
        "in_test_window": df["in_test_window"].resample(STEP).max(),
    })
    # Isi celah pendek (maks 30 menit = 3 titik) dengan interpolasi; celah yang lebih panjang tetap missing.
    r["soil"] = r["soil"].interpolate(limit=3, limit_area="inside")
    r["temp"] = r["temp"].interpolate(limit=3, limit_area="inside")
    r["missing"] = r["soil"].isna().astype(int)  # 1 = data hilang di slot ini
    r[["pump_on", "water_empty", "in_test_window"]] = r[["pump_on", "water_empty", "in_test_window"]].fillna(0)
    return r


def slope(series: pd.Series, window: int) -> pd.Series:
    """Kemiringan garis regresi (persen per jam) dalam jendela bergulir.

    Positif = kelembapan naik, negatif = kelembapan turun (tanah mengering).
    """
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum()

    def _s(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        # Rumus slope regresi linear sederhana, lalu dikali 6 agar satuannya per jam.
        return float(((y - y.mean()) * x).sum() / denom) * STEPS_PER_HOUR

    return series.rolling(window).apply(_s, raw=True)


def make_features(r: pd.DataFrame) -> pd.DataFrame:
    """Langkah 4: buat fitur deret waktu dari data 10 menit."""
    f = pd.DataFrame(index=r.index)
    s, t = r["soil"], r["temp"]
    f["soil_now"] = s  # kelembapan saat ini
    for h in (1, 3, 6, 12, 24):
        f[f"soil_lag_{h}h"] = s.shift(h * STEPS_PER_HOUR)          # nilai h jam yang lalu
        f[f"soil_diff_{h}h"] = s - s.shift(h * STEPS_PER_HOUR)     # perubahan selama h jam terakhir
    f["soil_mean_3h"] = s.rolling(3 * STEPS_PER_HOUR).mean()       # rata-rata 3 jam terakhir
    f["soil_std_3h"] = s.rolling(3 * STEPS_PER_HOUR).std()         # fluktuasi 3 jam terakhir
    f["soil_slope_6h"] = slope(s, 6 * STEPS_PER_HOUR)              # tren 6 jam (%/jam)
    f["soil_slope_24h"] = slope(s, 24 * STEPS_PER_HOUR)            # tren 24 jam (%/jam)
    f["temp_now"] = t
    f["temp_mean_6h"] = t.rolling(6 * STEPS_PER_HOUR, min_periods=12).mean()
    f["temp_diff_1h"] = t - t.shift(STEPS_PER_HOUR)
    # Jam diubah ke sin/cos supaya jam 23 dan jam 0 dianggap berdekatan (siklus harian).
    hour = r.index.hour + r.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Riwayat pompa: berapa kali menyala dalam 24 jam, dan berapa jam sejak terakhir menyala (maks 72).
    f["pump_count_24h"] = r["pump_on"].rolling(24 * STEPS_PER_HOUR, min_periods=1).sum()
    last_pump = r.index.to_series().where(r["pump_on"] > 0).ffill()
    hours_since = (r.index.to_series() - last_pump).dt.total_seconds() / 3600
    f["hours_since_pump"] = hours_since.fillna(72).clip(upper=72)
    return f


# Fitur model final: kondisi tanah sekarang + perubahan kelembapan 1, 3, 6, 12, dan 24 jam terakhir.
FEATURES = ["soil_now", "soil_diff_1h", "soil_diff_3h", "soil_diff_6h", "soil_diff_12h", "soil_diff_24h"]
# Fitur tambahan (suhu tanah dan jam) yang diuji tetapi tidak memperbaiki akurasi.
FEATURES_WIBH_TEMP = FEATURES + ["temp_now", "temp_diff_1h", "hour_sin", "hour_cos"]


def make_dataset(r: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Langkah 5: gabungkan fitur dengan target H jam ke depan, lalu buang sampel tidak valid."""
    k = horizon_h * STEPS_PER_HOUR      # horizon dalam jumlah langkah 10 menit
    f = make_features(r)
    future = r["soil"].shift(-k)        # nilai kelembapan H jam ke depan
    f["target_delta"] = future - r["soil"]   # target yang dipelajari model: PERUBAHAN kelembapan
    f["target_soil"] = future                # nilai absolut, untuk evaluasi

    # Jendela [t-24 jam, t+H] tidak boleh mengandung penyiraman, data hilang, atau periode uji sensor.
    # Alasannya: penyiraman membuat kelembapan melonjak tiba-tiba, bukan pola pengeringan alami.
    bad = ((r["pump_on"] > 0) | (r["missing"] > 0) | (r["in_test_window"] > 0)).astype(int)
    # bad_future: apakah ada kejadian "bad" dalam k langkah ke depan (rolling dibalik arah waktunya).
    bad_future = bad[::-1].rolling(k + 1, min_periods=1).max()[::-1]
    # bad_past: apakah ada data hilang dalam 24 jam ke belakang (fitur diff_24h jadi tidak andal).
    bad_past = r["missing"].rolling(24 * STEPS_PER_HOUR + 1, min_periods=1).max()
    f["valid"] = (bad_future == 0) & (bad_past == 0) & (r["in_test_window"] == 0)
    f = f[f["valid"]].dropna(subset=FEATURES_WIBH_TEMP + ["target_delta"])
    return f.drop(columns="valid")
