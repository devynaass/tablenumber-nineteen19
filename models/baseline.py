import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from skimage.feature import hog
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Tambahkan path direktori utama agar folder src terbaca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def buat_data_simulasi():
    """Membuat data simulasi citra 224x224 & target 8 kelas objek meja."""
    x_dummy = torch.rand(30, 3, 224, 224)  # 30 gambar simulasi
    y_dummy = torch.randint(0, 5, (30, 8)).float()  # 8 kelas jumlah objek
    dataset = TensorDataset(x_dummy, y_dummy)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    return train_loader, val_loader


def ekstrak_fitur_batch(dataloader):
    """Mengekstrak fitur Histogram Warna + HOG dari setiap gambar."""
    fitur_list = []
    target_list = []

    for images, targets in dataloader:
        # Ubah tensor (B, C, H, W) ke numpy (B, H, W, C) rentang 0-1
        imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
        imgs_np = np.clip(imgs_np, 0.0, 1.0)

        for img in imgs_np:
            # 1. Fitur Histogram Warna (16 bin per channel RGB = 48 fitur)
            hist_r, _ = np.histogram(img[:, :, 0], bins=16, range=(0, 1), density=True)
            hist_g, _ = np.histogram(img[:, :, 1], bins=16, range=(0, 1), density=True)
            hist_b, _ = np.histogram(img[:, :, 2], bins=16, range=(0, 1), density=True)
            color_hist = np.concatenate([hist_r, hist_g, hist_b])

            # 2. Fitur HOG (Histogram of Oriented Gradients)
            hog_feat = hog(
                img,
                orientations=8,
                pixels_per_cell=(32, 32),
                cells_per_block=(1, 1),
                channel_axis=-1,
            )

            # Gabungkan kedua fitur
            combined_feat = np.concatenate([color_hist, hog_feat])
            fitur_list.append(combined_feat)

        target_list.append(targets.cpu().numpy())

    X = np.array(fitur_list)
    y = np.vstack(target_list)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Pembanding Non-DL (Color Hist + HOG + RF)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== [Baseline Non-DL] Menjalankan Pembanding dengan Seed: {args.seed} ===")

    # 1. Muat Data (Otomatis pakai data primer jika Malika sudah selesai, atau simulasi jika belum)
    print("[1/4] Memuat data...")
    try:
        from src.protokol.data import muat
        train_loader, val_loader = muat()
        sumber_data = "data_primer"
    except (NotImplementedError, Exception):
        print("-> src/protokol/data.py belum siap, menggunakan data simulasi.")
        train_loader, val_loader = buat_data_simulasi()
        sumber_data = "simulasi"

    # 2. Ekstraksi Fitur Histogram Warna + HOG
    print("[2/4] Mengekstrak fitur Histogram Warna + HOG...")
    X_train, y_train = ekstrak_fitur_batch(train_loader)
    X_val, y_val = ekstrak_fitur_batch(val_loader)
    print(f"-> Dimensi fitur X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 3. Latih Model Random Forest Regressor
    print("[3/4] Melatih model Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 4. Evaluasi & Simpan ke results/baseline/metrics.json
    print("[4/4] Mengevaluasi dan menyimpan results/baseline/metrics.json...")
    y_pred = rf_model.predict(X_val)

    # Hitung MAE keseluruhan & MAE per kelas (8 kelas)
    mae_overall = float(mean_absolute_error(y_val, y_pred))
    mae_per_kelas = [float(mean_absolute_error(y_val[:, i], y_pred[:, i])) for i in range(8)]

    # Ketepatan susunan (exact match setelah pembulatan prediksi jumlah objek)
    y_pred_round = np.round(y_pred)
    exact_match = float(np.mean(np.all(y_pred_round == y_val, axis=1)))

    hasil_metrics = {
        "model": "baseline_non_dl_hist_hog_rf",
        "sumber_data": sumber_data,
        "seed": args.seed,
        "mae_overall": round(mae_overall, 4),
        "mae_per_kelas": [round(m, 4) for m in mae_per_kelas],
        "ketepatan_susunan": round(exact_match, 4),
    }

    # Pastikan folder results/baseline tersedia lalu simpan JSON
    os.makedirs("results/baseline", exist_ok=True)
    output_path = "results/baseline/metrics.json"
    with open(output_path, "w") as f:
        json.dump(hasil_metrics, f, indent=4)

    print(f"BERHASIL! Hasil disimpan di {output_path}:")
    print(json.dumps(hasil_metrics, indent=4))


if __name__ == "__main__":
    main()