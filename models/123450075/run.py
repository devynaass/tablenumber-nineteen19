import os
import sys
import time
import subprocess
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

# Tambahkan path direktori utama agar folder src terbaca
AKAR = Path(__file__).resolve().parents[2]
sys.path.append(str(AKAR))

from src.protokol.data import muat


def set_seed(seed: int):
    """Mengatur random seed agar eksperimen deterministik."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def siapkan_tensor(X_raw, y_raw=None):
    """Mengonversi array NumPy ke Tensor PyTorch (N, C, H, W) dengan normalisasi ImageNet."""
    X = np.array(X_raw, dtype=np.float32)
    # Pastikan urutan dimensi adalah (N, C, H, W)
    if X.ndim == 4 and X.shape[-1] in [1, 3, 4] and X.shape[1] not in [1, 3, 4]:
        X = np.transpose(X, (0, 3, 1, 2))
    if X.max() > 1.0:
        X = X / 255.0

    # Normalisasi standar ResNet (ImageNet mean & std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    if X.shape[1] == 3:
        X = (X - mean) / std

    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y_raw is not None:
        y_tensor = torch.tensor(np.array(y_raw), dtype=torch.float32)
        return X_tensor, y_tensor
    return X_tensor


class TransferLearningModel(nn.Module):
    """Model Transfer Learning (ResNet18) untuk regresi jumlah 8 objek meja."""

    def __init__(self, num_outputs=8):
        super(TransferLearningModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Buka pembekuan pada layer4 dan fc agar model bisa beradaptasi baik dengan data meja
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.backbone(x)


def prediksi_batch(model, X_tensor, device, batch_size=32):
    """Melakukan inferensi per batch dan mengembalikan array NumPy."""
    model.eval()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)


def main():
    parser = argparse.ArgumentParser(description="Latih Transfer Learning ResNet18 pada Data Primer")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch pelatihan")
    args = parser.parse_args()

    mulai_waktu = time.time()
    set_seed(args.seed)
    print(f"=== [TL ResNet18] Melatih pada Data Primer (Seed: {args.seed}) ===")

    # 1. Muat Data Primer dari src.protokol.data
    print("[1/4] Memuat data primer dari src.protokol.data...")
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = muat()
    print(f"-> Train: {X_train_raw.shape}, Val: {X_val_raw.shape}")

    X_train, y_train = siapkan_tensor(X_train_raw, y_train_raw)
    X_val, y_val = siapkan_tensor(X_val_raw, y_val_raw)

    num_outputs = y_train.shape[1] if y_train.ndim > 1 else 1
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    # 2. Siapkan Model, Loss (SmoothL1/MAE), dan Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perangkat_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/4] Melatih model ResNet18 selama {args.epochs} epoch di perangkat: {perangkat_str}...")

    model = TransferLearningModel(num_outputs=num_outputs).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_val_mae = float("inf")
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        # Evaluasi MAE pada data validasi tiap epoch
        val_preds = prediksi_batch(model, X_val, device)
        val_mae = float(np.mean(np.abs(val_preds - y_val_raw)))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/{args.epochs:02d} | Val MAE: {val_mae:.4f} (Terbaik: {best_val_mae:.4f})")

    # Gunakan bobot terbaik selama pelatihan
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 3. Simpan Prediksi Sementara & Panggil Modul Evaluasi Resmi Tim (src.common.evaluate)
    print("[3/4] Menghitung metrik resmi dan menulis results/123450075/metrics.json...")
    pred_val_final = prediksi_batch(model, X_val, device)

    temp_dir = AKAR / "results" / "123450075"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pred_val_path = temp_dir / "pred_val_tmp.npy"
    np.save(pred_val_path, pred_val_final)

    cmd = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", "123450075",
        "--model", "transfer_learning_resnet18",
        "--seed", str(args.seed),
        "--perangkat", perangkat_str,
        "--durasi", f"{time.time() - mulai_waktu:.2f}",
        "--pred-val", str(pred_val_path),
    ]

    if X_test_raw is not None and y_test_raw is not None:
        X_test = siapkan_tensor(X_test_raw)
        pred_test_final = prediksi_batch(model, X_test, device)
        pred_test_path = temp_dir / "pred_test_tmp.npy"
        np.save(pred_test_path, pred_test_final)
        cmd.extend(["--pred-test", str(pred_test_path)])

    subprocess.run(cmd, check=True)

    # Hapus file prediksi sementara agar repo tetap bersih
    if pred_val_path.exists():
        pred_val_path.unlink()
    if (temp_dir / "pred_test_tmp.npy").exists():
        (temp_dir / "pred_test_tmp.npy").unlink()

    print("[4/4] BERHASIL! Tugas #14 selesai dan results/123450075/metrics.json valid!")


if __name__ == "__main__":
    main()