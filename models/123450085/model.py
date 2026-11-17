"""models/123450085/model.py

Arsitektur CNN-LSTM (Jalur RNN) atas urutan strip horizontal citra 224x224.
"""

import torch
import torch.nn as nn


class StripCNN(nn.Module):
    """CNN kecil untuk mengekstrak fitur visual dari 1 strip baris citra."""

    def __init__(self, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNLSTM(nn.Module):
    """Model utama: Memotong citra 224x224 menjadi 7 strip horizontal,

    diekstrak CNN, lalu dibaca LSTM secara sekuensial dari atas ke bawah.
    """

    def __init__(
        self, num_classes=6, num_strips=7, feature_dim=128, hidden_dim=64
    ):
        super().__init__()
        self.num_strips = num_strips
        self.encoder = StripCNN(out_features=feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        B, C, H, W = x.shape
        strip_h = H // self.num_strips

        # Potong citra menjadi 7 strip baris horizontal
        strips = [
            x[:, :, i * strip_h : (i + 1) * strip_h, :]
            for i in range(self.num_strips)
        ]

        # Ekstrak fitur tiap strip
        features = [self.encoder(s) for s in strips]
        seq = torch.stack(features, dim=1)  # (B, 7, feature_dim)

        # Proses sekuensial dengan LSTM
        out, (hn, cn) = self.lstm(seq)

        # Klasifikasi kelas dari hidden state terakhir
        logits = self.fc(hn[-1])
        return logits