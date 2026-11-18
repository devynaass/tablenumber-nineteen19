
"""Satu perintah reproduksi dan komparasi Tim 19.

python run_all.py [--seeds 1 2 3] [--final] [--hanya NIM ...]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from src.common import manifest, split

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def baca_metrics(path: Path) -> dict | None:
    if not path.exists():
        print(f"[PERINGATAN] Tidak ditemukan: {path}")
        return None

    try:
        hasil = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[PERINGATAN] Gagal membaca {path}: {e}")
        return None

    mt = hasil.get("metrik")
    wajib = ("val_rmse", "val_mae", "val_r2")

    if not isinstance(mt, dict) or not all(
        isinstance(mt.get(k), (int, float)) and math.isfinite(mt[k])
        for k in wajib
    ):
        print(f"[PERINGATAN] Skema metrik regresi tidak valid: {path}")
        return None

    if not hasil.get("data_hash") or "seed" not in hasil:
        print(f"[PERINGATAN] data_hash/seed tidak tersedia: {path}")
        return None

    return hasil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None)
    ap.add_argument("--tanpa-baseline", action="store_true")
    ap.add_argument("--gunakan-hasil-ada", action="store_true")
    a = ap.parse_args()

    if not a.seeds:
        ap.error("Minimal satu seed harus diberikan.")
        
    hash_acuan = manifest.data_hash(
        f"split:{split.SEED_SPLIT}".encode()
)

    if not a.tanpa_baseline:
        jalankan([
            sys.executable, "-m", "src.common.baseline",
            "--seed", str(a.seeds[0])
        ])

    anggota = sorted(
        p for p in (AKAR / "models").iterdir()
        if p.is_dir()
        and (p / "run.py").exists()
        and p.name != "TEMPLATE"
        and (a.hanya is None or p.name in a.hanya)
    )

    ringkas: dict[str, list[dict]] = {}

    for model in anggota:
        for seed in a.seeds:
            perintah = [
                sys.executable, str(model / "run.py"),
                "--seed", str(seed)
            ]

            if a.final and seed == a.seeds[-1]:
                perintah.append("--final")

            if a.gunakan_hasil_ada:
                print(f"[PAKAI HASIL ADA] {model.name}, seed {seed}")
            else:
                jalankan(perintah)

            path = AKAR / "results" / model.name / "metrics.json"
            hasil = baca_metrics(path)

            if hasil is not None:
                ringkas.setdefault(model.name, []).append(hasil)

    baris = [
        "| NIM | Model | Seed | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² | Durasi (s) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    ]

    def angka(nilai):
        return f"{nilai:.4f}" if isinstance(nilai, (int, float)) else "—"

    if not a.tanpa_baseline:
        path = AKAR / "results" / "baseline" / "metrics.json"
        hasil = baca_metrics(path)
        if hasil:
            mt = hasil["metrik"]
            status = (
            "valid"
        if hasil.get("data_hash") == hash_acuan
            else "hash tidak cocok"
)
            baris.append(
                f"| baseline | {hasil.get('model', 'baseline')} "
                f"| {hasil.get('seed', '—')} "
                f"| {angka(mt.get('val_rmse'))} "
                f"| {angka(mt.get('val_mae'))} "
                f"| {angka(mt.get('val_r2'))} "
                f"| {angka(mt.get('test_rmse'))} "
                f"| {angka(mt.get('test_mae'))} "
                f"| {angka(mt.get('test_r2'))} "
                f"| {hasil.get('durasi_detik', 0):.1f} |  {status} |"
            )
        else:
            baris.append(
                "| baseline | — | — | — | — | — | — | — | — | — | tidak valid |"
            )

    for nim, daftar in ringkas.items():
        for hasil in daftar:
            mt = hasil["metrik"]

            status = (
                "valid"
                if hasil.get("data_hash") == hash_acuan
                else "hash tidak cocok"
            )
            baris.append(
                f"| {nim} | {hasil.get('model', '—')} "
                f"| {hasil.get('seed', '—')} "
                f"| {angka(mt.get('val_rmse'))} "
                f"| {angka(mt.get('val_mae'))} "
                f"| {angka(mt.get('val_r2'))} "
                f"| {angka(mt.get('test_rmse'))} "
                f"| {angka(mt.get('test_mae'))} "
                f"| {angka(mt.get('test_r2'))} "
                f"| {hasil.get('durasi_detik', 0):.1f} | {status} |"
            )

    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text(
        "# Hasil Komparasi Sprint 1\n\n"
        "Metrik regresi dihitung dari berkas metrics.json. "
        "Tanda — berarti metrik tidak tersedia.\n\n"
        + "\n".join(baris) + "\n",
        encoding="utf-8"
    )

    print("Tabel komparasi →", out.relative_to(AKAR))


if __name__ == "__main__":
    main()

