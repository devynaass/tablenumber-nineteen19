"""models/123450085/run.py

Skrip eksekusi pelatihan dan evaluasi model CNN-LSTM (Jalur RNN).
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pastikan folder model bisa diimpor
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from model import CNNLSTM


def muat_dataset():
    """Memuat data dari data/simulasi yang sudah disiapkan."""
    sim_dir = "data/simulasi"
    if not os.path.exists(os.path.join(sim_dir, "X_train.npy")):
        raise FileNotFoundError(
            "Data simulasi belum ada! Jalankan 'python src/simulasi_data.py' terlebih dahulu."
        )

    X_tr = np.load(os.path.join(sim_dir, "X_train.npy"))
    y_tr = np.load(os.path.join(sim_dir, "y_train.npy"))
    X_va = np.load(os.path.join(sim_dir, "X_val.npy"))
    y_va = np.load(os.path.join(sim_dir, "y_val.npy"))
    return X_tr, y_tr, X_va, y_va


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # 1. Pasang Seed untuk reproduktibilitas
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 55)
    print("   MEMULAI RUN PIPELINE CNN-LSTM (JALUR RNN)    ")
    print("=" * 55)

    # 2. Muat Data
    X_tr, y_tr, X_va, y_va = muat_dataset()
    print(f"Data Latih     : X = {X_tr.shape}, y = {y_tr.shape}")
    print(f"Data Validasi  : X = {X_va.shape}, y = {y_va.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3. Inisialisasi Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    model = CNNLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. Pelatihan
    print(
        f"\nMelatih model pada {device} ({args.epochs} epoch, seed {args.seed})..."
    )
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f" Epoch [{epoch+1}/{args.epochs}] -> Loss: {total_loss/len(train_loader):.4f}"
        )

    durasi = time.time() - start_time
    print(f"Pelatihan selesai dalam {durasi:.2f} detik.")

    # 5. Inferensi Data Validasi
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)

    p_val = np.array(val_preds)

    # 6. Simpan Prediksi Validasi & Metrik
    out_dir = "results/123450085"
    os.makedirs(out_dir, exist_ok=True)
    pred_file = os.path.join(out_dir, "pred_val.npy")
    np.save(pred_file, p_val)

    # Hitung akurasi sederhana
    acc = float((p_val == y_va).mean())
    metrics_file = os.path.join(out_dir, "metrics.json")
    hasil = {
        "metrik": {"val_acc": round(acc, 4)},
        "seed": args.seed,
        "model": "rnn",
        "durasi_detik": round(durasi, 2),
        "perangkat": "cpu laptop" if device.type == "cpu" else "cuda gpu",
        "n_val": len(y_va),
    }
    with open(metrics_file, "w") as f:
        json.dump(hasil, f, indent=2)

    print("\n" + "=" * 55)
    print("HASIL EVALUASI PIPELINE:")
    print(f" - File Prediksi : {pred_file}")
    print(f" - File Metrik   : {metrics_file}")
    print(f" - Akurasi Val   : {acc * 100:.2f}%")
    print("=" * 55)
    print("STATUS: BERJALAN SAMPAI AKHIR DENGAN SUKSES!")


if __name__ == "__main__":
    main()