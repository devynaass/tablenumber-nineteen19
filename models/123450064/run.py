
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
MODEL = "trf"
JUMLAH_OUTPUT = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PatchTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            3, 64, kernel_size=16, stride=16
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 196, 64)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, JUMLAH_OUTPUT)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def normalisasi(X_train, X_val):
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--perangkat", type=str, default="cpu laptop"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(4)

    device = torch.device("cpu")

    print("Memuat dataset...")
    X_train, y_train, X_val, y_val, _, _ = data.muat()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    X_train, X_val = normalisasi(X_train, X_val)

    print("Train:", X_train.shape, y_train.shape)
    print("Validasi:", X_val.shape, y_val.shape)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = PatchTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    mulai = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        rata_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- Train Loss: {rata_loss:.6f}"
        )

    model.eval()

    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val).to(device)
        pred_val = model(val_tensor).cpu().numpy()

    durasi = time.time() - mulai

    hasil_dir = AKAR / "results" / NIM
    hasil_dir.mkdir(parents=True, exist_ok=True)

    pred_path = hasil_dir / "pred_val.npy"
    np.save(pred_path, pred_val)

    mae_per_kelas = np.mean(
        np.abs(y_val - pred_val),
        axis=0
    )

    pred_bulat = np.rint(
        np.clip(pred_val, 0, None)
    )

    exact_match = np.mean(
        np.all(pred_bulat == y_val, axis=1)
    )

    detail = {
        "mae_per_kelas": {
            nama: float(nilai)
            for nama, nilai in zip(
                data.KELAS_OBJEK,
                mae_per_kelas
            )
        },
        "exact_match_accuracy": float(exact_match),
        "jumlah_data_validasi": int(len(y_val)),
        "seed": args.seed
    }

    detail_path = hasil_dir / "metrics_detail.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    perintah = [
        sys.executable,
        "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", MODEL,
        "--seed", str(args.seed),
        "--pred-val", str(pred_path),
        "--durasi", str(durasi),
        "--perangkat", args.perangkat
    ]

    print("\nMenjalankan evaluator resmi...")
    subprocess.run(perintah, cwd=AKAR, check=True)

    print("\nTraining selesai.")
    print("Durasi:", round(durasi, 2), "detik")
    print("MAE per kelas:", detail["mae_per_kelas"])
    print("Exact match:", round(exact_match, 4))
    print("Hasil:", hasil_dir)


if __name__ == "__main__":
    main()
