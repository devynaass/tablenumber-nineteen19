"""run.py jalur Transformer (patch ViT kecil). Salin ke models/<nim>/run.py.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "trf"                                   # cnn | tl | rnn | trf | ae

UKURAN = 224    # masukan model tim: citra 224x224
PATCH = 16      # 224/16 = 14 -> 14x14 = 196 patch
DIM = 64
BATCH = 32
EPOCH = 2


def ke_nchw(X, torch):
    """Ubah batch data apa pun jadi tensor (B, C, H, W) float 0..1, lalu ke UKURAN x UKURAN."""
    t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if t.ndim == 2:                                   # data datar (N, H*W)
        s = int(round(t.shape[1] ** 0.5))
        if s * s != t.shape[1]:
            raise ValueError(f"Tidak bisa menebak bentuk citra dari {tuple(t.shape)}")
        t = t.reshape(len(t), 1, s, s)
    elif t.ndim == 3:                                 # (N, H, W)
        t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
        t = t.permute(0, 3, 1, 2)                     # (N, H, W, C) -> (N, C, H, W)
    if t.max() > 1.5:                                 # skala 0..255 -> 0..1
        t = t / 255.0
    if t.shape[-1] != UKURAN or t.shape[-2] != UKURAN:
        t = torch.nn.functional.interpolate(t, size=(UKURAN, UKURAN), mode="bilinear", align_corners=False)
    return t


def buat_model(in_ch: int, n_kelas: int, torch):
    nn = torch.nn

    class PatchTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Conv2d(in_ch, DIM, PATCH, PATCH)           # potong jadi patch + proyeksi
            self.pos = nn.Parameter(torch.zeros(1, (UKURAN // PATCH) ** 2, DIM))
            lapis = nn.TransformerEncoderLayer(DIM, 4, DIM * 2, batch_first=True)
            self.enc = nn.TransformerEncoder(lapis, 2)
            self.kepala = nn.Linear(DIM, n_kelas)

        def forward(self, x):                                          # (B, C, 224, 224)
            z = self.embed(x).flatten(2).transpose(1, 2) + self.pos    # (B, 196, DIM)
            return self.kepala(self.enc(z).mean(1))                    # (B, n_kelas)

    return PatchTransformer()


def prediksi(model, X, torch) -> np.ndarray:
    model.eval()
    hasil = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            hasil.append(model(ke_nchw(X[i:i + BATCH], torch)).argmax(1).numpy())
    return np.concatenate(hasil)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    ytr = np.asarray(ytr)
    n_kelas = int(ytr.max()) + 1
    in_ch = ke_nchw(Xtr[:1], torch).shape[1]

    model = buat_model(in_ch, n_kelas, torch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # tulis alasan pilihan optimizer di catatan.md
    y = torch.as_tensor(ytr).long()
    for _ in range(EPOCH):
        model.train()
        urutan = np.random.permutation(len(Xtr))
        for i in range(0, len(urutan), BATCH):
            idx = urutan[i:i + BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(ke_nchw(np.asarray(Xtr)[idx], torch)), y[idx])
            loss.backward()
            opt.step()

    pred_val = prediksi(model, Xva, torch)
    pred_test = prediksi(model, Xte, torch) if final and Xte is not None else None
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
