"""Cek manual citra dan label (bukan alat resmi)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402
from src.common import manifest  # noqa: E402


def nchw(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] != 3 and X.shape[-1] == 3:
        X = X.transpose(0, 3, 1, 2)
    return X


def ringkas(nama, X, y):
    print(f"[{nama}] citra {X.shape} dtype={X.dtype}  label {y.shape}")
    print(f"[{nama}] piksel min={X.min():.3f} maks={X.max():.3f} rerata={X.mean():.3f}")
    print(f"[{nama}] NaN citra={int(np.isnan(X).sum())}  NaN label={int(np.isnan(y).sum())}")
    print(f"[{nama}] label min={y.min():.0f} maks={y.max():.0f}  total objek/citra: "
          f"min={y.sum(1).min():.0f} maks={y.sum(1).max():.0f}")


def main():
    Xtr, ytr, Xva, yva, *_ = data.muat()
    Xtr, Xva = nchw(Xtr), nchw(Xva)
    ytr, yva = np.asarray(ytr, np.float32), np.asarray(yva, np.float32)

    ringkas("latih", Xtr, ytr)
    ringkas("val  ", Xva, yva)
    print(f"data_hash: {manifest.data_hash()}  (metrics.json memakai hash dengan parameter split)")
    kelas = list(getattr(data, "KELAS_OBJEK", [f"kelas{i}" for i in range(ytr.shape[1])]))
    print("kelas:", kelas)
    print("rata-rata jumlah per kelas (latih):", dict(zip(kelas, np.round(ytr.mean(0), 2))))

    idx = np.linspace(0, len(Xva) - 1, 8).astype(int)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 4, figsize=(16, 9))
        for a, i in zip(ax.ravel(), idx):
            a.imshow(np.clip(Xva[i].transpose(1, 2, 0), 0, 1))
            isi = ", ".join(f"{k}:{int(v)}" for k, v in zip(kelas, yva[i]) if v > 0)
            a.set_title(f"val #{i}\n{isi}", fontsize=8)
            a.axis("off")
        out = AKAR / "docs" / "cek_citra_cnn.png"
        fig.tight_layout()
        fig.savefig(out, dpi=80)
        print(f"montase disimpan: {out.relative_to(AKAR)}")
    except ImportError:
        print("matplotlib tidak ada, montase dilewati")


if __name__ == "__main__":
    main()
