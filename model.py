"""
Module du modèle de détection de deepfakes.
Architecture : CNN + Frequency + rPPG + Transformer
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNFreqRPPGTransformer(nn.Module):
    """
    Architecture multimodale pour la détection de deepfakes.
    Combine : CNN (ResNet-18), features de fréquence, signal rPPG, Transformer.
    """
    
    def __init__(self, freq_dim: int = 10, rppg_dim: int = 6,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        # ── 1. CNN backbone (ResNet-18) ──────────────────────────────
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        cnn_out = 512

        self.cnn_proj = nn.Linear(cnn_out, d_model)
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── 2. Cross-Attention A : freq → CNN ────────────────────────
        self.cross_attn_A = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm_A_q = nn.LayerNorm(d_model)
        self.norm_A_kv = nn.LayerNorm(d_model)
        self.norm_A_out = nn.LayerNorm(d_model)
        self.ffn_A = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_A_ffn = nn.LayerNorm(d_model)

        # ── 3. Transformer Encoder temporel ──────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 4. Cross-Attention B : rPPG → zt ────────────────────────
        self.rppg_proj = nn.Sequential(
            nn.Linear(rppg_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn_B = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm_B_q = nn.LayerNorm(d_model)
        self.norm_B_kv = nn.LayerNorm(d_model)
        self.norm_B_out = nn.LayerNorm(d_model)
        self.ffn_B = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_B_ffn = nn.LayerNorm(d_model)

        # Gate
        self.gate_B = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # ── 5. Classifier ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, frames: torch.Tensor,
                freq: torch.Tensor,
                rppg: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, 3, H, W)
        freq   : (B, T, freq_dim)
        rppg   : (B, rppg_dim)
        Retourne : (B, 2) logits [real, fake]
        """
        B, T, C, H, W = frames.shape

        # ── 1. CNN + freq projections ────────────────────────────────
        cnn_in = frames.view(B * T, C, H, W)
        cnn_out = self.cnn(cnn_in).flatten(1)
        cnn_out = self.cnn_proj(cnn_out).view(B, T, -1)

        freq_out = self.freq_proj(freq)

        # ── 2. Cross-Attention A : freq (Q) → CNN (K, V) ────────────
        q_A = self.norm_A_q(freq_out)
        kv_A = self.norm_A_kv(cnn_out)

        attn_A, _ = self.cross_attn_A(
            query=q_A, key=kv_A, value=kv_A
        )

        tokens_A = self.norm_A_out(freq_out + attn_A)
        tokens_A = self.norm_A_ffn(tokens_A + self.ffn_A(tokens_A))

        # ── 3. Transformer Encoder temporel ─────────────────────────
        zt = self.temporal_encoder(tokens_A)
        pooled = zt.mean(dim=1)

        # ── 4. Cross-Attention B : rPPG (Q) → zt (K, V) ─────────────
        rppg_emb = self.rppg_proj(rppg)
        q_B = self.norm_B_q(rppg_emb).unsqueeze(1)
        kv_B = self.norm_B_kv(zt)

        attn_B, _ = self.cross_attn_B(
            query=q_B, key=kv_B, value=kv_B
        )
        attn_B = attn_B.squeeze(1)

        attended = self.norm_B_out(rppg_emb + attn_B)
        attended = self.norm_B_ffn(attended + self.ffn_B(attended))

        # Gate
        g = self.gate_B(torch.cat([attended, pooled], dim=-1))
        output = g * attended + (1.0 - g) * pooled

        # ── 5. Classify ─────────────────────────────────────────────
        return self.head(output)


def load_model(checkpoint_path: str, device: str = "cpu") -> CNNFreqRPPGTransformer:
    """Charge un modèle entraîné depuis un checkpoint."""
    model = CNNFreqRPPGTransformer(
        freq_dim=10, rppg_dim=6,
        d_model=256, nhead=8, num_layers=3,
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model
