import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, Sequential, Linear, ReLU
import math
from copy import deepcopy

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

# ========== SINGLE ENCODER MODEL ==========

class TransformerEncoderLayerWithAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # place to store the weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # same as parent forward, except we intercept attn_weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        if not self.training:
            self.last_attn_weights = attn_weights.detach()  # store for later

        # rest of the computation is copied from TransformerEncoderLayer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerWithAttn

# Modular single encoder
class SEModel(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        self.seq_len = grid_length + grid_length
        
        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length + (self.condition_dim is not None), d_model, device
        )
        self.full_pos = torch.cat([self.shared_pos[:, :(self.grid_length + (self.condition_dim is not None)), :],
                            self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEModel

# ======================== FiLM ==============================

class FiLMAdapter(nn.Module):
    def __init__(self, d_model, guidance_dim):
        super().__init__()
        self.gamma = nn.Linear(guidance_dim, d_model)
        self.beta = nn.Linear(guidance_dim, d_model)

        # neutralize FiLM
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
    # end init

    def forward(self, x, guidance):
        """
        x: (B, L, d_model)
        guidance: (B, guidance_dim)
        """
        gamma = self.gamma(guidance).unsqueeze(1)  # (B, 1, d_model)
        beta = self.beta(guidance).unsqueeze(1)    # (B, 1, d_model)

        # Identity-safe initialization expectation:
        # gamma initialized near 1, beta near 0
        return gamma * x + beta
    # end forwad
# end FiLMAdapter

class TransformerEncoderLayerWithFiLM(nn.Module):
    def __init__(self, encoder_layer, d_model, guidance_dim=None):
        super().__init__()
        self.layer = encoder_layer
        self.guidance_dim = guidance_dim

        if guidance_dim is not None:
            self.film = FiLMAdapter(d_model, guidance_dim)
        else:
            self.film = None

        self.last_attn_weights = None
    # end init

    def forward(self, src, guidance=None, src_mask=None, src_key_padding_mask=None):
        # --- Standard TransformerEncoderLayer forward ---
        src2, attn_weights = self.layer.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        if not self.training:
            self.last_attn_weights = attn_weights.detach()

        src = src + self.layer.dropout1(src2)
        src = self.layer.norm1(src)
        src2 = self.layer.linear2(
            self.layer.dropout(
                self.layer.activation(self.layer.linear1(src))
            )
        )
        src = src + self.layer.dropout2(src2)
        src = self.layer.norm2(src)

        # --- FiLM conditioning ---
        if self.film is not None and guidance is not None:
            src = self.film(src, guidance)

        return src
    # end forward
# end TransformerEncoderLayerWithFiLM

class SEFiLMModel(nn.Module):
    def __init__(
        self,
        chord_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        dropout=0.3,
        guidance_dim=None,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.guidance_dim = guidance_dim

        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        self.seq_len = grid_length * 2

        self.shared_pos = sinusoidal_positional_encoding(
            self.seq_len, d_model, device
        )

        self.dropout = nn.Dropout(dropout)

        # --- Encoder Layers with FiLM ---
        self.encoder_layers = nn.ModuleList()

        for _ in range(num_layers):
            base_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder_layers.append(
                TransformerEncoderLayerWithFiLM(
                    base_layer,
                    d_model=d_model,
                    guidance_dim=guidance_dim
                )
            )

        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.to(device)
    # end init

    def forward(
        self,
        melody_grid,
        harmony_tokens=None,
        guidance_embedding=None,
        return_hidden=False
    ):
        B = melody_grid.size(0)
        device = self.device

        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(
                B, self.grid_length, self.d_model, device=device
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        full_seq = full_seq + self.shared_pos
        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        encoded = full_seq

        for layer in self.encoder_layers:
            encoded = layer(encoded, guidance=guidance_embedding)

        encoded = self.output_norm(encoded)

        harmony_logits = self.output_head(
            encoded[:, -self.grid_length:, :]
        )

        if return_hidden:
            return harmony_logits, encoded
        else:
            return harmony_logits
    # end forward

    def freeze_base(self):
        for name, param in self.named_parameters():
            if "film" not in name:
                param.requires_grad = False
    # end freeze_base

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # end unfreeze_all

# end SEFiLMModel