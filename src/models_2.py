import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        rnn_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        
        self.rnn = nn.LSTM(
            input_size = hid_dim,
            hidden_size = rnn_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional=True
        )

        self.attention = AttentionBlock(rnn_dim * 2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(rnn_dim * 2, num_classes),
        )

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)  # (b, hid_dim, t)
        X = X.permute(0, 2, 1)  # (b, t, hid_dim) for RNN
        X, _ = self.rnn(X)  # (b, t, rnn_dim * 2)

        X = self.attention(X)  # (b, rnn_dim * 2)
        X = X.unsqueeze(-1)  # (b, rnn_dim * 2) -> (b, rnn_dim * 2, 1)
        
        return self.head(X)  # (b, rnn_dim * 2, 1) -> (b, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X  # Skip connection

        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.dropout(X)

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))
        X = self.dropout(X)

        X += residual  # Skip connection

        return X

class AttentionBlock(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X, _ = self.attention(X, X, X)
        X = self.norm(X)
        return X.mean(dim=1)  # (b, input_dim)

