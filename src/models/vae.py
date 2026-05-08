import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import N_PITCHES, SEQ_LEN


class Encoder(nn.Module):
    def __init__(self, input_dim=N_PITCHES, hidden_dim=256, latent_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm       = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.dropout    = nn.Dropout(dropout)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, 88) → mu, log_var: (B, latent_dim)
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=N_PITCHES,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        # z: (B, latent_dim) → (B, T, 88) raw logits
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.fc(self.dropout(out))


class MusicVAE(nn.Module):
    def __init__(self, input_dim=N_PITCHES, hidden_dim=256, latent_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder    = Decoder(latent_dim, hidden_dim, input_dim, num_layers, dropout)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparametrize(mu, log_var)
        return self.decoder(z, x.size(1)), mu, log_var   # logits, mu, log_var

    @staticmethod
    def focal_loss(logits, targets, gamma=2.0, pos_weight=16.0):
        bce          = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs        = torch.sigmoid(logits)
        p_t          = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t).pow(gamma)
        class_weight = targets * pos_weight + (1 - targets)
        return (focal_weight * class_weight * bce).mean()

    @staticmethod
    def kl_loss(mu, log_var):
        # closed form: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    @torch.no_grad()
    def generate(self, n=8, seq_len=SEQ_LEN, device='cpu', temperature=1.0, threshold=0.3):
        self.eval()
        z      = torch.randn(n, self.latent_dim, device=device) * temperature
        logits = self.decoder(z, seq_len)
        return (torch.sigmoid(logits) > threshold).float()   # (n, T, 88)

    @torch.no_grad()
    def interpolate(self, x1, x2, steps=8, seq_len=SEQ_LEN, threshold=0.3):
        # x1, x2: (T, 88) single samples → zα = (1-α)μ1 + α·μ2 for α ∈ linspace(0,1,steps)
        self.eval()
        mu1, _ = self.encoder(x1.unsqueeze(0))
        mu2, _ = self.encoder(x2.unsqueeze(0))
        rolls  = []
        for a in torch.linspace(0, 1, steps, device=mu1.device):
            z    = (1 - a) * mu1 + a * mu2
            roll = (torch.sigmoid(self.decoder(z, seq_len)) > threshold).float()
            rolls.append(roll)
        return torch.cat(rolls, dim=0)   # (steps, T, 88)
