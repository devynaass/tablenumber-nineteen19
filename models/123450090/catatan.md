
# Catatan Model Autoencoder

**NIM:** 123450090

**Nama model:** Convolutional Autoencoder + Regression Head

## 1. Tujuan
Menggunakan Autoencoder untuk mempelajari representasi fitur citra meja dan memprediksi jumlah delapan kelas objek.

## 2. Arsitektur
- CNN Encoder untuk ekstraksi fitur.
- Latent representation berukuran 128.
- CNN Decoder untuk rekonstruksi citra.
- Regression Head untuk prediksi delapan jumlah objek.

## 3. Fungsi Loss
Total loss merupakan gabungan:
- Reconstruction Loss: Mean Squared Error (MSE).
- Prediction Loss: Smooth L1 Loss.

## 4. Optimizer
Adam dengan learning rate 0.001.

## 5. Konfigurasi Awal
- Epoch: 10
- Batch size: 16
- Input: 224 × 224 piksel
- Output: 8 nilai prediksi jumlah objek.
- Perangkat: CPU laptop.

## 6. Pengujian
Pengujian awal direncanakan menggunakan data simulasi.

## 7. Catatan
Konfigurasi dan arsitektur dapat disesuaikan setelah pengujian simulasi dan persetujuan tim.
