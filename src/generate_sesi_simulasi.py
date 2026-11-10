"""
src/generate_sesi_simulasi.py

Generator Sesi Simulasi 120 Kondisi & Kartu Susunan Resmi Tim 19
Berdasarkan spesifikasi t19-objek-meja.yaml:
- 30 susunan (S01 s.d. S30)
- 8 kelas objek: buku, botol, ponsel, gunting, spidol, gelas, kotak pensil, kunci
- 2 taraf lampu (0, 255)
- 2 ulangan (R1, R2)
Total: 30 x 2 x 2 = 120 kondisi

Menghasilkan:
1. data/simulasi/sesi_t19/label.csv (120 kondisi untuk Penjaga Data)
2. data/simulasi/sesi_t19/manifest.json
3. docs/kartu/label_30_susunan.csv (30 susunan resmi)
4. docs/kartu/kartu_susunan_t19.pdf (kartu susunan cetak S01-S30)
5. docs/kartu/checklist_latihan_kering.md (panduan checklist latihan kering)
"""

import os
import re
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(ROOT_DIR, "t19-objek-meja.yaml")
SESI_DIR = os.path.join(ROOT_DIR, "data", "simulasi", "sesi_t19")
KARTU_DIR = os.path.join(ROOT_DIR, "docs", "kartu")

os.makedirs(SESI_DIR, exist_ok=True)
os.makedirs(KARTU_DIR, exist_ok=True)

KELAS = [
    "buku", "botol", "ponsel", "gunting",
    "spidol", "gelas", "kotak pensil", "kunci"
]

def parse_yaml():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    taraf = cfg["faktor"]["susunan"]["taraf"]
    lampu_list = cfg["faktor"]["lampu"]
    ulangan = cfg.get("ulangan", 2)

    # 1. Parse 30 susunan
    susunan_list = []
    for item in taraf:
        code, desc = item.split(": ")
        counts = {c: 0 for c in KELAS}
        for part in desc.split(","):
            part = part.strip()
            m = re.match(r"^(\d+)\s+(.+)$", part)
            if m:
                qty = int(m.group(1))
                name = m.group(2).strip()
            else:
                qty = 1
                name = part
            for c in KELAS:
                if name.lower() == c.lower():
                    counts[c] = qty

        total = sum(counts.values())
        susunan_list.append({
            "susunan": code,
            "deskripsi": desc,
            **counts,
            "total_objek": total
        })

    # 2. Bangun 120 kondisi
    kondisi_list = []
    kondisi_idx = 1
    for s in susunan_list:
        for l in lampu_list:
            for u in range(1, ulangan + 1):
                k_id = f"K{kondisi_idx:03d}_{s['susunan']}_L{l}_R{u}"
                kondisi_list.append({
                    "id_kondisi": k_id,
                    "susunan": s["susunan"],
                    "lampu": l,
                    "ulangan": u,
                    **{c: s[c] for c in KELAS},
                    "total_objek": s["total_objek"]
                })
                kondisi_idx += 1

    return cfg, susunan_list, kondisi_list

def generate_files():
    cfg, susunan_list, kondisi_list = parse_yaml()

    # Simpan label_30_susunan.csv
    df_30 = pd.DataFrame(susunan_list)
    path_30 = os.path.join(KARTU_DIR, "label_30_susunan.csv")
    df_30.to_csv(path_30, index=False)
    print(f"[OK] 30 susunan tersimpan di: {path_30}")

    # Simpan label.csv (120 kondisi)
    df_120 = pd.DataFrame(kondisi_list)
    path_120 = os.path.join(SESI_DIR, "label.csv")
    df_120.to_csv(path_120, index=False)
    print(f"[OK] 120 kondisi sesi tersimpan di: {path_120}")

    # Simpan manifest.json
    manifest_data = {
        "tim": cfg.get("tim", 19),
        "judul": cfg.get("judul", "Deteksi dan hitung objek di meja"),
        "total_kondisi": len(kondisi_list),
        "total_susunan": len(susunan_list),
        "kelas_objek": KELAS,
        "faktor": {
            "lampu": cfg["faktor"]["lampu"],
            "ulangan": cfg.get("ulangan", 2)
        },
        "kamera": cfg.get("rekam", {}).get("kamera", {}),
        "file_label": "label.csv"
    }
    path_manifest = os.path.join(SESI_DIR, "manifest.json")
    with open(path_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[OK] Manifest sesi tersimpan di: {path_manifest}")

    # Buat PDF kartu_susunan_t19.pdf (A4 bersih untuk dicetak/dilihat saat latihan)
    pdf_path = os.path.join(KARTU_DIR, "kartu_susunan_t19.pdf")
    with PdfPages(pdf_path) as pdf:
        # Tampilkan 2 kartu per halaman A4
        for i in range(0, len(susunan_list), 2):
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
            for j in range(2):
                idx = i + j
                ax = axes[j]
                if idx < len(susunan_list):
                    item = susunan_list[idx]
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    ax.axis("off")

                    # Bingkai kartu
                    rect = plt.Rectangle((0.5, 0.5), 9, 9, fill=False, edgecolor="#1f77b4", linewidth=2.5, linestyle="-")
                    ax.add_patch(rect)

                    # Header Kartu
                    ax.text(5, 8.6, f"KARTU SUSUNAN {item['susunan']}", fontsize=18, fontweight="bold", ha="center", color="#0b3954")
                    ax.text(5, 8.0, "Tim 19: Deteksi & Hitung Objek di Meja (Area A3)", fontsize=10, ha="center", color="#555555")
                    ax.plot([1.2, 8.8], [7.7, 7.7], color="#cccccc", lw=1.5)

                    # Daftar Objek
                    ax.text(1.5, 7.0, f"Daftar Objek: {item['deskripsi']}", fontsize=13, fontweight="bold", color="#2c3e50")
                    ax.text(1.5, 6.3, f"Total Objek: {item['total_objek']} buah", fontsize=11, fontstyle="italic", color="#2980b9")

                    # Rincian 8 Kelas dalam Bentuk Grid Teks
                    rincian_lines = []
                    for c in KELAS:
                        qty = item[c]
                        if qty > 0:
                            rincian_lines.append(f"• {c.title()}: {qty}")
                    rincian_text = "   |   ".join(rincian_lines)
                    ax.text(1.5, 5.2, "Rincian Kelas Aktif:", fontsize=10, fontweight="bold", color="#333333")
                    ax.text(1.5, 4.5, rincian_text, fontsize=10, color="#444444")

                    # Checklist Aturan Penataan
                    ax.text(1.5, 3.2, "Aturan Penataan Meja A3:", fontsize=10, fontweight="bold", color="#333333")
                    ax.text(1.7, 2.5, "[  ] Tidak saling menumpuk secara fisik\n[  ] Semua objek di dalam batas kertas A3\n[  ] Letak berbeda dari susunan sebelumnya", fontsize=9, color="#555555")

                    # Footer
                    ax.text(5, 1.0, f"Slot Pengambilan Data Rabu 11 Nov — Kartu #{item['susunan']}", fontsize=8, ha="center", color="#888888")
                else:
                    ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[OK] Kartu PDF resmi tersimpan di: {pdf_path}")

    # Buat Checklist Latihan Kering Markdown
    path_check = os.path.join(KARTU_DIR, "checklist_latihan_kering.md")
    with open(path_check, "w", encoding="utf-8") as f:
        f.write("# Checklist Latihan Kering Menata Susunan (Tim 19)\n\n")
        f.write("**Penanggung Jawab:** Juesi Apridelia Saragih (123450085)\n\n")
        f.write("Gunakan checklist ini saat latihan menata objek di atas kertas A3 sesuai urutan kartu S01 s.d. S30:\n\n")
        f.write("| Kartu | Susunan Objek | Total | Tidak Menumpuk | Dalam Batas A3 | Posisi Beda | Status |\n")
        f.write("|:---:|---|:---:|:---:|:---:|:---:|:---:|\n")
        for s in susunan_list:
            f.write(f"| **{s['susunan']}** | {s['deskripsi']} | {s['total_objek']} | [ ] | [ ] | [ ] | Pending |\n")
        f.write("\n> *Centang `[x]` setiap kali Anda selesai mempraktikkan susunan fisik di atas kertas A3.*\n")
    print(f"[OK] Checklist latihan kering tersimpan di: {path_check}")

if __name__ == "__main__":
    generate_files()
