import os
import sys
import json
import argparse
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Tambahkan path direktori utama agar folder src terbaca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.protokol.data import muat


def ekstrak_fitur_numpy(X_array):
    """Mengekstrak fitur Histogram Warna + HOG dari array gambar NumPy."""
    fitur_list = []

    for img in X_array:
        # Pastikan urutan dimensi adalah (H, W, C) jika awalnya (C, H, W)
        if img.ndim == 3 and img.shape[0] in [1, 3, 4] and img.shape[2] not in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))

        # Normalisasi rentang piksel ke 0.0 - 1.0
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)

        # Jika gambar masih 1D (sudah di-flatten), gunakan langsung
        if img.ndim == 1:
            fitur_list.append(img)
            continue

        # 1. Fitur Histogram Warna (16 bin per channel RGB = 48 fitur)
        if img.ndim == 3 and img.shape[2] >= 3:
            hist_r, _ = np.histogram(img[:, :, 0], bins=16, range=(0, 1), density=True)
            hist_g, _ = np.histogram(img[:, :, 1], bins=16, range=(0, 1), density=True)
            hist_b, _ = np.histogram(img[:, :, 2], bins=16, range=(0, 1), density=True)
            color_hist = np.concatenate([hist_r, hist_g, hist_b])
            hog_feat = hog(
                img,
                orientations=8,
                pixels_per_cell=(32, 32),
                cells_per_block=(1, 1),
                channel_axis=-1,
            )
        else:
            color_hist, _ = np.histogram(img, bins=16, range=(0, 1), density=True)
            hog_feat = hog(
                img.squeeze(),
                orientations=8,
                pixels_per_cell=(32, 32),
                cells_per_block=(1, 1),
            )

        combined_feat = np.concatenate([color_hist, hog_feat])
        fitur_list.append(combined_feat)

    return np.array(fitur_list)


def main():
    parser = argparse.ArgumentParser(description="Pembanding Non-DL (Color Hist + HOG + RF)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"=== [Baseline Non-DL] Menjalankan Pembanding dengan Seed: {args.seed} ===")

    # 1. Muat Data Primer dari src/protokol/data.py buatan Malika
    print("[1/4] Memuat data primer dari src.protokol.data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = muat()
    print(f"-> Data dimuat! Train: {X_train_raw.shape}, Val: {X_val_raw.shape}")

    # 2. Ekstraksi Fitur Histogram Warna + HOG
    print("[2/4] Mengekstrak fitur Histogram Warna + HOG...")
    X_train = ekstrak_fitur_numpy(X_train_raw)
    X_val = ekstrak_fitur_numpy(X_val_raw)
    print(f"-> Dimensi fitur X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 3. Latih Model Random Forest Regressor
    print("[3/4] Melatih model Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 4. Evaluasi & Simpan ke results/baseline/metrics.json
    print("[4/4] Mengevaluasi dan menyimpan results/baseline/metrics.json...")
    y_pred = rf_model.predict(X_val)

    num_classes = y_val.shape[1] if y_val.ndim > 1 else 1
    mae_overall = float(mean_absolute_error(y_val, y_pred))

    if y_val.ndim > 1:
        mae_per_kelas = [float(mean_absolute_error(y_val[:, i], y_pred[:, i])) for i in range(num_classes)]
        y_pred_round = np.round(y_pred)
        exact_match = float(np.mean(np.all(y_pred_round == y_val, axis=1)))
    else:
        mae_per_kelas = [mae_overall]
        exact_match = float(np.mean(np.round(y_pred) == y_val))

    hasil_metrics = {
        "model": "baseline_non_dl_hist_hog_rf",
        "sumber_data": "data_primer",
        "seed": args.seed,
        "mae_overall": round(mae_overall, 4),
        "mae_per_kelas": [round(m, 4) for m in mae_per_kelas],
        "ketepatan_susunan": round(exact_match, 4),
    }

    import subprocess, sys
    os.makedirs("results/baseline", exist_ok=True)
    np.save("results/baseline/pred_val.npy", y_pred)
    with open("results/baseline/metrics_detail.json", "w", encoding="utf-8") as f:
        json.dump(hasil_metrics, f, indent=2)
    subprocess.run([
        sys.executable, "-m", "src.common.evaluate",
        "--nim", "baseline",
        "--model", "baseline",
        "--seed", str(args.seed),
        "--pred-val", "results/baseline/pred_val.npy",
        "--durasi", "1.5",
        "--perangkat", "cpu laptop"
    ], check=True)


if __name__ == "__main__":
    main()
