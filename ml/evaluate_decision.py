"""
Membandingkan keputusan sistem AI dengan sistem rule-based Tugas 2 pada periode uji (4-12 Nov 2026).

- Rule-based  : tanah dianggap butuh air jika kelembapan sekarang < 55%.
- AI (hybrid) : tanah dianggap akan kering jika peluang kelembapan 12 jam lagi < 55% >= 0,70
                (peluang dihitung dari prediksi Linear Regression dan sigma galat walk-forward),
                ditambah aturan lama sebagai pengaman.
Label kebenaran: kelembapan aktual turun di bawah 55% dalam 12 jam berikutnya.
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prepare import DRY_THRESHOLD, FEATURES, STEPS_PER_HOUR, clean, load_raw, make_dataset, make_features, resample  # noqa: E402

# ---------------------------------------------------------------- konfigurasi
ROOT = Path(__file__).resolve().parents[1]
H = 12              # horizon prediksi yang dipakai untuk keputusan (jam)
P_THRESHOLD = 0.70  # AI memberi peringatan jika peluang kering >= 70%
TEST_DAYS = pd.date_range("2026-11-04", "2026-11-12", freq="D")


def prob_below(pred: np.ndarray, sigma: float, threshold: float = DRY_THRESHOLD) -> np.ndarray:
    """Peluang kelembapan aktual < threshold, jika galat prediksi dianggap berdistribusi normal.

    Rumus CDF normal: P = 0.5 * (1 + erf((threshold - pred) / (sigma * sqrt(2)))).
    Contoh: pred = 55 (tepat di batas) -> P = 0.5; pred jauh di bawah 55 -> P mendekati 1.
    """
    z = (threshold - pred) / (sigma * sqrt(2))
    return 0.5 * (1 + np.vectorize(erf)(z))


def main():
    # 1. Siapkan data (sama seperti train.py) dan baca sigma dari model final hasil train.py.
    raw = clean(load_raw(ROOT / "data" / "dataset.csv"))
    r = resample(raw)
    ds = make_dataset(r, H)       # dataset latih (hanya sampel valid)
    feats = make_features(r)      # fitur untuk SEMUA titik waktu (dipakai saat memprediksi)
    model_info = json.loads((ROOT / "results" / "soil_forecast_model.json").read_text())
    sigma = model_info["horizons"][str(H)]["sigma"]

    # 2. Prediksi walk-forward pada SETIAP titik uji yang fiturnya lengkap (hanya butuh data masa lalu).
    #    Berbeda dengan train.py, di sini titik yang terkena penyiraman tetap diprediksi, karena
    #    dashboard di dunia nyata juga harus membuat keputusan di titik-titik tersebut.
    parts = []
    for day in TEST_DAYS:
        train = ds[ds.index < day - pd.Timedelta(hours=H)]           # data sebelum hari uji
        lr = LinearRegression().fit(train[FEATURES], train["target_delta"])
        x = feats[(feats.index >= day) & (feats.index < day + pd.Timedelta(days=1))].dropna(subset=FEATURES)
        if x.empty:
            continue
        pred = (x["soil_now"] + lr.predict(x[FEATURES])).clip(0, 100)
        parts.append(pd.DataFrame({"soil_now": x["soil_now"], "pred_12h": pred}, index=x.index))
    rep = pd.concat(parts)
    # 3. Ubah prediksi menjadi peluang tanah kering 12 jam lagi.
    rep["p_dry"] = prob_below(rep["pred_12h"].to_numpy(), sigma)

    # 4. Label kebenaran: 1 jika kelembapan aktual minimum dalam (t, t+12 jam] < 55%.
    #    Rolling dibalik ([::-1]) agar menghitung minimum ke DEPAN, lalu shift(-1) agar titik t sendiri tidak ikut.
    k = H * STEPS_PER_HOUR
    future_min = r["soil"][::-1].rolling(k, min_periods=k // 2).min()[::-1].shift(-1)
    rep["future_min"] = future_min.reindex(rep.index)
    rep = rep.dropna(subset=["future_min"])
    rep["label"] = (rep["future_min"] < DRY_THRESHOLD).astype(int)
    # 5. Keputusan kedua sistem:
    #    rule : aturan lama Tugas 2, hanya melihat kelembapan sekarang.
    #    ai   : peluang kering >= 70% ATAU aturan lama (aturan lama tetap jadi pengaman).
    rep["rule"] = (rep["soil_now"] < DRY_THRESHOLD).astype(int)
    rep["ai"] = ((rep["p_dry"] >= P_THRESHOLD) | (rep["soil_now"] < DRY_THRESHOLD)).astype(int)

    # 6. Confusion matrix dan metrik klasifikasi untuk masing-masing sistem.
    #    TP = benar memperingatkan, FP = alarm palsu, FN = terlambat/terlewat, TN = benar diam.
    rows = []
    for name in ("rule", "ai"):
        y, yhat = rep["label"], rep[name]
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        rows.append({
            "sistem": "Rule-based (Tugas 2)" if name == "rule" else "AI hybrid (Final Project)",
            "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    res = pd.DataFrame(rows)

    # 7. Waktu peringatan pertama sebelum tanah benar-benar kering (selisihnya = lead time AI).
    first_dry = r.index[(r.index >= "2026-11-04") & (r["soil"] < DRY_THRESHOLD)].min()
    first_ai = rep.index[rep["ai"] == 1].min()
    first_rule = rep.index[rep["rule"] == 1].min()
    raw_first_auto = raw.index[(raw["status"] == "MULAI AUTO") & (raw.index >= "2026-11-04")].min()

    # 8. Kinerja aturan lama pada 5-6 Nov: berapa kali pompa menyala dan bagaimana kelembapan berubah.
    win = raw.loc["2026-11-05":"2026-11-06 06:00"]
    starts = win[win["status"] == "MULAI AUTO"]
    rule_stats = {
        "auto_starts": int(len(starts)),
        "first_start": str(starts.index.min()), "last_start": str(starts.index.max()),
        "soil_at_first_start": int(starts["soil_percent"].iloc[0]),
        "soil_at_last_start": int(starts["soil_percent"].iloc[-1]),
        "pump_seconds_total": int(len(starts) * 8),
    }
    # 9. False alarm pada siklus kedua (7-12 Nov), saat tanah tidak pernah kering; lalu simpan semua hasil.
    late = rep.loc["2026-11-07":]
    summary = {
        "test_points": len(rep), "positives": int(rep["label"].sum()),
        "first_actual_below_55": str(first_dry), "first_rule_flag": str(first_rule),
        "first_ai_flag": str(first_ai), "raw_first_mulai_auto": str(raw_first_auto),
        "lead_time_hours_vs_rule": round((first_rule - first_ai).total_seconds() / 3600, 2),
        "ai_alerts_7_12_nov": int(late["ai"].sum()), "rule_alerts_7_12_nov": int(late["rule"].sum()),
        "points_7_12_nov": len(late), "rule_based_5_6_nov": rule_stats,
        "sigma_12h": sigma, "p_threshold": P_THRESHOLD,
    }
    res.to_csv(ROOT / "results" / "decision_comparison.csv", index=False)
    rep.to_csv(ROOT / "results" / "decision_replay.csv", index_label="timestamp_wit")
    (ROOT / "results" / "decision_summary.json").write_text(json.dumps(summary, indent=2))
    print(res.round(3).to_string()); print(json.dumps(summary, indent=2))

    # 10. Grafik timeline keputusan di sekitar kejadian tanah kering pertama.
    #     Garis hijau = waktu peringatan AI pertama, garis merah = waktu aturan lama pertama aktif.
    seg = rep.loc["2026-11-04":"2026-11-06 06:00"]
    act = r.loc["2026-11-04":"2026-11-06 06:00", "soil"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(act.index, act, color="#2d3748", lw=1, label="Kelembapan aktual")
    ax.plot(seg.index + pd.Timedelta(hours=H), seg["pred_12h"], color="#2f855a", lw=1, ls=":", label="Prediksi 12 jam (digeser ke waktu target)")
    ax.axhline(DRY_THRESHOLD, ls="--", lw=0.8, color="gray")
    ax.axvline(first_ai, color="#2f855a", lw=1.2)
    ax.axvline(first_rule, color="#c53030", lw=1.2)
    ax.text(first_ai, 92, f"peringatan AI pertama  \n{first_ai:%d %b %H:%M}  ", color="#2f855a", fontsize=8, va="top", ha="right")
    ax.text(first_rule, 84, f"  aturan lama (< 55%)\n  {first_rule:%d %b %H:%M}", color="#c53030", fontsize=8, va="top")
    ax.set_ylim(30, 95); ax.set_ylabel("Kelembapan (%)")
    ax.set_title("Waktu peringatan: AI vs rule-based (4–6 Nov 2026)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "fig" / "fig_decision_timeline.png", dpi=170); plt.close(fig)

    # 11. Prediksi 12 jam vs aktual untuk seluruh periode uji (titik prediksi digeser ke waktu target).
    fig, ax = plt.subplots(figsize=(10, 4))
    act_all = r.loc["2026-11-04":, "soil"]
    ax.plot(act_all.index, act_all, color="#2d3748", lw=1, label="Kelembapan aktual")
    ax.plot(rep.index + pd.Timedelta(hours=H), rep["pred_12h"], ".", ms=2, color="#2f855a", label="Prediksi 12 jam sebelumnya")
    ax.axhline(DRY_THRESHOLD, ls="--", lw=0.8, color="gray", label="Batas kering 55%")
    ax.set_ylabel("Kelembapan (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, markerscale=4, loc="lower right")
    ax.set_title("Prediksi 12 jam ke depan vs kelembapan aktual, periode uji 4–12 Nov 2026")
    fig.tight_layout(); fig.savefig(ROOT / "fig" / "fig_pred_vs_actual.png", dpi=170); plt.close(fig)

    # 12. Confusion matrix kedua sistem (warna makin gelap = jumlah makin besar).
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for a, row in zip(axes, rows):
        m = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        a.imshow(m, cmap="Greens")
        for i in range(2):
            for j in range(2):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=11)
        a.set_xticks([0, 1], ["Pred: aman", "Pred: kering"]); a.set_yticks([0, 1], ["Aktual: aman", "Aktual: kering"])
        a.set_title(row["sistem"], fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "fig" / "fig_confusion.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
