
"""
Convolutional Autoencoder + Regression Head
Anggota: 123450090
Jalur model: AE

Menjalankan:
    python models/123450090/run.py --seed 1

Evaluasi data uji hanya pada sprint terakhir:
    python models/123450090/run.py --seed 1 --final
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# KONFIGURASI PROYEK
# ============================================================

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

NIM = "123450090"
JALUR = "ae"

DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

JUMLAH_KELAS = 8
UKURAN_CITRA = 224


# ============================================================
# PREPROCESSING
# ============================================================

def siapkan_citra(X: np.ndarray) -> np.ndarray:
    """
    Mengubah citra menjadi format:
    [N, C, 224, 224]

    Mendukung input:
    [N, H, W]
    [N, H, W, C]
    [N, C, H, W]
    """

    X = np.asarray(X)

    if X.ndim == 3:
        # [N, H, W] -> [N, 1, H, W]
        X = X[:, None, :, :]

    elif X.ndim == 4:
        # Mengenali format NHWC
        if X.shape[-1] in (1, 3, 4):
            X = np.transpose(X, (0, 3, 1, 2))

        # Jika RGBA, gunakan RGB
        if X.shape[1] == 4:
            X = X[:, :3, :, :]

    else:
        raise ValueError(
            f"Bentuk citra tidak didukung: {X.shape}"
        )

    X = X.astype(np.float32)

    # Mengubah nilai piksel 0-255 menjadi 0-1
    if X.max() > 1.0:
        X = X / 255.0

    if X.min() < 0 or X.max() > 1:
        raise ValueError(
            "Nilai citra harus berada pada rentang 0 sampai 1."
        )

    if X.shape[1] not in (1, 3):
        raise ValueError(
            f"Jumlah channel harus 1 atau 3, ditemukan {X.shape[1]}"
        )

    return X


def siapkan_label(y: np.ndarray) -> np.ndarray:
    """
    Target regresi berupa delapan jumlah objek.
    Bentuk: [N, 8]
    """

    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        if y.size % JUMLAH_KELAS != 0:
            raise ValueError(
                "Label harus memiliki delapan nilai per sampel."
            )
        y = y.reshape(-1, JUMLAH_KELAS)

    if y.ndim != 2 or y.shape[1] != JUMLAH_KELAS:
        raise ValueError(
            f"Target harus berbentuk [N, 8], ditemukan {y.shape}"
        )

    if not np.isfinite(y).all():
        raise ValueError("Label mengandung NaN atau infinity.")

    return y


# ============================================================
# ARSITEKTUR CONVOLUTIONAL AUTOENCODER
# ============================================================

class AutoencoderRegresi(nn.Module):

    def __init__(self, channel: int = 3):
        super().__init__()

        # ---------------- ENCODER ----------------
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(channel, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        # ---------------- DECODER ----------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(128, 128 * 14 * 14),
            nn.ReLU(inplace=True),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, 4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, 4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32, 4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, channel, 4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

        # ---------------- REGRESSION HEAD ----------------
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, JUMLAH_KELAS),
            nn.Softplus(),
        )

    def forward(self, x):

        fitur = self.encoder_cnn(x)
        latent = self.encoder_fc(fitur)

        hasil_decoder = self.decoder_fc(latent)
        hasil_decoder = hasil_decoder.reshape(
            -1, 128, 14, 14
        )

        rekonstruksi = self.decoder_cnn(hasil_decoder)

        prediksi_jumlah = self.regression_head(latent)

        return rekonstruksi, prediksi_jumlah


# ============================================================
# TRAINING DAN PREDIKSI
# ============================================================

def latih_dan_prediksi(
    seed: int,
    final: bool
) -> tuple[np.ndarray, np.ndarray | None]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Memanggil pemuat data milik tim
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Evaluator saat ini mendukung regresi untuk target hitungan.
    if getattr(data, "TUGAS", None) != "regresi":
        raise RuntimeError(
            "Model AE ini menghasilkan delapan nilai hitungan "
            "per citra. TUGAS pada src/protokol/data.py harus "
            "disepakati sebagai 'regresi' oleh tim terlebih dahulu."
        )

    Xtr = siapkan_citra(Xtr)
    Xva = siapkan_citra(Xva)

    ytr = siapkan_label(ytr)
    yva = siapkan_label(yva)

    if len(Xtr) != len(ytr):
        raise ValueError("Jumlah X_train dan y_train tidak sama.")

    if len(Xva) != len(yva):
        raise ValueError("Jumlah X_val dan y_val tidak sama.")

    if Xtr.shape[1:] != Xva.shape[1:]:
        raise ValueError("Bentuk citra train dan validation berbeda.")

    if Xtr.shape[-2:] != (UKURAN_CITRA, UKURAN_CITRA):
        raise ValueError(
            "Citra harus berukuran 224 x 224. "
            f"Ditemukan {Xtr.shape[-2:]}."
        )

    # Dataset PyTorch
    dataset = TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Inisialisasi model
    model = AutoencoderRegresi(
        channel=Xtr.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_rekonstruksi = nn.MSELoss()
    loss_regresi = nn.SmoothL1Loss()

    # ---------------- TRAINING ----------------
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            rekonstruksi, prediksi = model(xb)

            loss_ae = loss_rekonstruksi(
                rekonstruksi, xb
            )

            loss_count = loss_regresi(
                prediksi, yb
            )

            # Kombinasi loss rekonstruksi dan prediksi jumlah
            loss = loss_ae + loss_count

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        rata_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {rata_loss:.6f}"
        )

    # ---------------- VALIDATION ----------------
    model.eval()

    with torch.no_grad():

        xv = torch.from_numpy(Xva).to(DEVICE)

        _, pred_val = model(xv)

        pred_val = pred_val.cpu().numpy()

    # ---------------- TEST ----------------
    pred_test = None

    if final and Xte is not None and yte is not None:

        Xte = siapkan_citra(Xte)

        if Xte.shape[1:] != Xtr.shape[1:]:
            raise ValueError(
                "Bentuk citra test tidak sama dengan train."
            )

        model.eval()

        with torch.no_grad():

            xt = torch.from_numpy(Xte).to(DEVICE)

            _, hasil_test = model(xt)

            pred_test = hasil_test.cpu().numpy()

    return pred_val, pred_test


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluasi data uji pada sprint terakhir."
    )

    parser.add_argument(
        "--perangkat",
        default="cpu laptop"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("AUTOENCODER - 123450090")
    print("=" * 55)

    mulai = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        seed=args.seed,
        final=args.final
    )

    durasi = time.time() - mulai

    # Simpan prediksi
    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    path_val = output / "pred_val.npy"
    np.save(path_val, pred_val)

    print(f"Prediksi validasi disimpan: {path_val}")

    # Perintah evaluator standar tim
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(path_val),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", args.perangkat,
    ]

    # Data test hanya digunakan saat --final
    if pred_test is not None:

        path_test = output / "pred_test.npy"

        np.save(path_test, pred_test)

        perintah += [
            "--pred-test",
            str(path_test)
        ]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )

    print("=" * 55)
    print("PROSES SELESAI")
    print(f"Durasi: {durasi:.2f} detik")
    print(f"Hasil: {output}")
    print("=" * 55)


if __name__ == "__main__":
    main()
