import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, Sequential, Linear, ReLU
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
from collections import defaultdict
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

# ========== DUAL ENCODER MODEL ==========

# ========== Small helper: Transformer encoder layer with cross-attention ==========
class HarmonyEncoderLayerWithCross(nn.Module):
    """
    One layer for harmony encoder:
      - self-attn (harmony -> harmony)
      - cross-attn (harmony queries -> melody keys/values)
      - feed-forward
    Stores last attention weights for diagnostics.
    """
    def __init__(
                self,
                d_model, 
                nhead, 
                dim_feedforward=2048, 
                dropout=0.3, 
                activation='gelu', 
                batch_first=True,
                device='cpu'
            ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)

        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff_in = nn.Dropout(dropout)
        self.dropout_ff_out = nn.Dropout(dropout)

        # placeholders for attention visualization
        self.last_self_attn = None  # shape (B, nhead, Lh, Lh) if requested
        self.last_cross_attn = None # shape (B, nhead, Lh, Lm) if requested
    # end init

    def forward(self, x_h, melody_kv, attn_mask=None, key_padding_mask=None, melody_key_padding_mask=None):
        """
        x_h: (B, Lh, d_model) harmony input
        melody_kv: (B, Lm, d_model) melody encoded (keys & values)
        attn_mask: optional for self-attn
        key_padding_mask: optional for self-attn
        melody_key_padding_mask: optional for cross-attn (for melody padding)
        """
        # Self-attention
        x_h = self.norm1(x_h) # pre norm
        h2, self_w = self.self_attn(x_h, x_h, x_h,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        # self_w : (B, nhead, Lh, Lh)  if batch_first and average_attn_weights=False
        if not self.training:
            self.last_self_attn = self_w.detach() if isinstance(self_w, torch.Tensor) else None

        x_h = x_h + self.dropout1(h2)
        # x_h = self.norm1(x_h)
        # x_h = self.norm1(h2)

        # Cross-attention: queries = harmony (x_h), keys/values = melody_kv
        x_h = self.norm2(x_h) # pre norm
        c2, cross_w = self.cross_attn(x_h, melody_kv, melody_kv,
                                      key_padding_mask=melody_key_padding_mask,
                                      need_weights=True,
                                      average_attn_weights=False)
        if not self.training:
            self.last_cross_attn = cross_w.detach() if isinstance(cross_w, torch.Tensor) else None

        x_h = x_h + self.dropout2(c2)
        # x_h = self.norm2(x_h)
        # x_h = self.norm2(c2)

        # Feed-forward
        x_h = self.norm3(x_h) # pre norm
        ff = self.linear2(self.dropout_ff_in(self.activation(self.linear1(x_h))))
        x_h = x_h + self.dropout_ff_out(ff)
        # x_h = self.norm3(x_h)

        return x_h
    # end forward
# end class HarmonyEncoderLayerWithCross

class HarmonyEncoderLayerOnlyCross(nn.Module):
    """
    One layer for harmony encoder:
      - cross-attn (harmony queries -> melody keys/values)
      - feed-forward
    Stores last attention weights for diagnostics.
    """
    def __init__(
                self,
                d_model, 
                nhead, 
                dim_feedforward=2048, 
                dropout=0.3, 
                activation='gelu', 
                batch_first=True,
                device='cpu'
            ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)

        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff_in = nn.Dropout(dropout)
        self.dropout_ff_out = nn.Dropout(dropout)

        # placeholders for attention visualization
        # self.last_self_attn = None  # shape (B, nhead, Lh, Lh) if requested
        self.last_cross_attn = None # shape (B, nhead, Lh, Lm) if requested
    # end init

    def forward(self, x_h, melody_kv, attn_mask=None, key_padding_mask=None, melody_key_padding_mask=None):
        """
        x_h: (B, Lh, d_model) harmony input
        melody_kv: (B, Lm, d_model) melody encoded (keys & values)
        attn_mask: optional for self-attn
        key_padding_mask: optional for self-attn
        melody_key_padding_mask: optional for cross-attn (for melody padding)
        """
        # Self-attention
        x_h = self.norm1(x_h) # pre norm
        # h2, self_w = self.self_attn(x_h, x_h, x_h,
        #                             attn_mask=attn_mask,
        #                             key_padding_mask=key_padding_mask,
        #                             need_weights=True,
        #                             average_attn_weights=False)
        # # self_w : (B, nhead, Lh, Lh)  if batch_first and average_attn_weights=False
        # if not self.training:
        #     self.last_self_attn = self_w.detach() if isinstance(self_w, torch.Tensor) else None

        # x_h = x_h + self.dropout1(h2)
        # x_h = self.norm1(x_h)
        # x_h = self.norm1(h2)

        # Cross-attention: queries = harmony (x_h), keys/values = melody_kv
        x_h = self.norm2(x_h) # pre norm
        c2, cross_w = self.cross_attn(x_h, melody_kv, melody_kv,
                                      key_padding_mask=melody_key_padding_mask,
                                      need_weights=True,
                                      average_attn_weights=False)
        if not self.training:
            self.last_cross_attn = cross_w.detach() if isinstance(cross_w, torch.Tensor) else None

        x_h = x_h + self.dropout2(c2)
        # x_h = self.norm2(x_h)
        # x_h = self.norm2(c2)

        # Feed-forward
        x_h = self.norm3(x_h) # pre norm
        ff = self.linear2(self.dropout_ff_in(self.activation(self.linear1(x_h))))
        x_h = x_h + self.dropout_ff_out(ff)
        # x_h = self.norm3(x_h)

        return x_h
    # end forward
# end class HarmonyEncoderLayerOnlyCross

# ========== Stacked encoder modules ==========
class SimpleTransformerStack(nn.Module):
    """A small wrapper that stacks standard nn.TransformerEncoderLayer layers (no cross-attn)."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    # end init

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, L, D) assumed batch_first=True
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
    # end forward
# end class SimpleTransformerStack

class HarmonyTransformerStack(nn.Module):
    """Stack of HarmonyEncoderLayerWithCross"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    # end init

    def forward(self, x_h, melody_kv, h_key_padding_mask=None, melody_key_padding_mask=None):
        # x_h: (B, Lh, D)
        for layer in self.layers:
            x_h = layer(x_h, melody_kv,
                        key_padding_mask=h_key_padding_mask,
                        melody_key_padding_mask=melody_key_padding_mask)
        return x_h
    # end forward
# end class HarmonyTransformerStack

# ========== Dual-encoder model ==========
class DualGridMLMMelHarm(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            max(melody_length, harmony_length), d_model, device
        )

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = TransformerEncoderLayerWithAttn(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerWithCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm_mel = nn.LayerNorm(d_model, device=device)
        self.output_norm_mel = nn.LayerNorm(d_model, device=device)
        self.input_norm_harm = nn.LayerNorm(d_model, device=device)
        self.output_norm_harm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None, *args, **kwargs):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos[:, :self.melody_length, :]
        mel = self.input_norm_mel(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=None)  # (B, Lm, d_model)
        mel_encoded = self.output_norm_mel(mel_encoded)

        # ---- Harmony embedding ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.shared_pos[:, :self.harmony_length, :]

        harm = self.input_norm_harm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded)  # (B, Lh, d_model)

        harm_encoded = self.output_norm_harm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            self_attns.append(layer.last_self_attn)
            cross_attns.append(layer.last_cross_attn)
        return self_attns, cross_attns
    # end get_attention_maps
# end class DualGridMLMMelHarm

class DE_learned_pos(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            melody_length, d_model, device
        )
        self.learned_pos = nn.Parameter(torch.zeros(1, harmony_length, d_model, device=device))

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = TransformerEncoderLayerWithAttn(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerWithCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm_mel = nn.LayerNorm(d_model, device=device)
        self.output_norm_mel = nn.LayerNorm(d_model, device=device)
        self.input_norm_harm = nn.LayerNorm(d_model, device=device)
        self.output_norm_harm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None, *args, **kwargs):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos
        mel = self.input_norm_mel(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=None)  # (B, Lm, d_model)
        mel_encoded = self.output_norm_mel(mel_encoded)

        # ---- Harmony embedding ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.learned_pos

        harm = self.input_norm_harm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded)  # (B, Lh, d_model)

        harm_encoded = self.output_norm_harm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            self_attns.append(layer.last_self_attn)
            cross_attns.append(layer.last_cross_attn)
        return self_attns, cross_attns
    # end get_attention_maps
# end class DE_learned_pos

class DE_only_cross(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            max(melody_length, harmony_length), d_model, device
        )

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = TransformerEncoderLayerWithAttn(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerOnlyCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm_mel = nn.LayerNorm(d_model, device=device)
        self.output_norm_mel = nn.LayerNorm(d_model, device=device)
        self.input_norm_harm = nn.LayerNorm(d_model, device=device)
        self.output_norm_harm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None, *args, **kwargs):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos[:, :self.melody_length, :]
        mel = self.input_norm_mel(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=None)  # (B, Lm, d_model)
        mel_encoded = self.output_norm_mel(mel_encoded)

        # ---- Harmony embedding ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.shared_pos[:, :self.harmony_length, :]

        harm = self.input_norm_harm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded)  # (B, Lh, d_model)

        harm_encoded = self.output_norm_harm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            cross_attns.append(layer.last_cross_attn)
        return cross_attns
    # end get_attention_maps
# end class DE_only_cross

class DE_no_MHself(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            max(melody_length, harmony_length), d_model, device
        )

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = TransformerEncoderLayerΝοAttn(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerOnlyCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm_mel = nn.LayerNorm(d_model, device=device)
        self.output_norm_mel = nn.LayerNorm(d_model, device=device)
        self.input_norm_harm = nn.LayerNorm(d_model, device=device)
        self.output_norm_harm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None, *args, **kwargs):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos[:, :self.melody_length, :]
        mel = self.input_norm_mel(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=None)  # (B, Lm, d_model)
        mel_encoded = self.output_norm_mel(mel_encoded)

        # ---- Harmony embedding ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.shared_pos[:, :self.harmony_length, :]

        harm = self.input_norm_harm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded)  # (B, Lh, d_model)

        harm_encoded = self.output_norm_harm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            cross_attns.append(layer.last_cross_attn)
        return cross_attns
    # end get_attention_maps
# end class DE_no_MHself

class DE_no_Mself(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            max(melody_length, harmony_length), d_model, device
        )

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = TransformerEncoderLayerΝοAttn(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerWithCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm_mel = nn.LayerNorm(d_model, device=device)
        self.output_norm_mel = nn.LayerNorm(d_model, device=device)
        self.input_norm_harm = nn.LayerNorm(d_model, device=device)
        self.output_norm_harm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None, *args, **kwargs):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos[:, :self.melody_length, :]
        mel = self.input_norm_mel(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=None)  # (B, Lm, d_model)
        mel_encoded = self.output_norm_mel(mel_encoded)

        # ---- Harmony embedding ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.shared_pos[:, :self.harmony_length, :]

        harm = self.input_norm_harm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded)  # (B, Lh, d_model)

        harm_encoded = self.output_norm_harm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            self_attns.append(layer.last_self_attn)
            cross_attns.append(layer.last_cross_attn)
        return self_attns, cross_attns
    # end get_attention_maps
# end class DE_no_Mself

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

class TransformerEncoderLayerΝοAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):

        # rest of the computation is copied from TransformerEncoderLayer
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerΝοAttn

class SingleGridMLMelHarm(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=4, 
                 num_layers=4, 
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 grid_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 1 + grid_length + grid_length # condition + melody + harmony
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        
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
        full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                              self.shared_pos[:, :self.grid_length, :]], dim=1)

        # Add positional encoding
        full_seq = full_seq + full_pos

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
# end class SingleGridMLMelHarm

# Modular single encoder
class SEModular(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            condition_dim=None,  # if not None, add a condition token of this dim at start
            unmasking_stages=None,  # if not None, use stage-based unmasking
            trainable_pos_emb=False,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.condition_dim = condition_dim
        self.unmasking_stages = unmasking_stages
        self.trainable_pos_emb = trainable_pos_emb

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        # If using condition token, a linear projection
        if self.condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, d_model, device=self.device)
            self.seq_len = 1 + grid_length + grid_length
        else:
            self.seq_len = grid_length + grid_length
        
        # Positional embeddings
        if self.trainable_pos_emb:
            self.full_pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model, device=device))
            nn.init.trunc_normal_(self.full_pos, std=0.02)
        else:
            # # Positional embeddings (separate for clarity)
            self.shared_pos = sinusoidal_positional_encoding(
                grid_length + (self.condition_dim is not None), d_model, device
            )
            self.full_pos = torch.cat([self.shared_pos[:, :(self.grid_length + (self.condition_dim is not None)), :],
                              self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # If using unmasking stages, add an embedding layer
        if self.unmasking_stages is not None:
            assert isinstance(self.unmasking_stages, int) and self.unmasking_stages > 0, "unmasking_stages must be a positive integer"
            self.stage_embedding_dim = 64
            self.stage_embedding = nn.Embedding(self.unmasking_stages, self.stage_embedding_dim, device=self.device)
            # New projection layer to go from (d_model + stage_embedding_dim) → d_model
            self.stage_proj = nn.Linear(self.d_model + self.stage_embedding_dim, self.d_model, device=self.device)
        
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

    def forward(self, melody_grid, harmony_tokens=None, conditioning_vec=None, stage_indices=None):
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
        if conditioning_vec is not None and self.condition_dim is not None:
            # Project condition: (B, d_model) → (B, 1, d_model)
            cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)
            full_seq = torch.cat([cond_emb, full_seq], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        if self.unmasking_stages is not None:
            # add stage embedding to harmony part
            stage_emb = self.stage_embedding(stage_indices)  # (B, stage_embedding_dim)
            stage_emb = stage_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, stage_embedding_dim)
            # Concatenate along the feature dimension
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)  # (B, seq_len, d_model + stage_embedding_dim)
            # Project back to d_model
            full_seq = self.stage_proj(full_seq)  # (B, seq_len, d_model)

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
# end class SEModular

# from simple models

# ======== Transformer Blocks ========

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, device='cpu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.last_attn = None  # store for visualization

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn_weights  # (batch, heads, seq, seq)
        return out
# end SelfAttention

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, device='cpu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.last_attn = None

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        out, attn_weights = self.attn(
            q, kv, kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn_weights  # (batch, heads, q_len, kv_len)
        return out
# end CrossAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, device='cpu'):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, device=device),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, device=device),
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.last_attn_weights = None  # place to store the weights
    # end init

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.last_attn_weights = self.self_attn.last_attn.detach()  # store for later
        x = self.norm1(x + sa)
        # Feedforward
        ff = self.ff(x)
        x = self.norm2(x + ff)
        return x
# end TransformerBlock

class CrossTransformerBlock(nn.Module):
    """For H-encoder with self + cross attention"""
    def __init__(self, d_model, n_heads, dim_ff, device='cpu'):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, device=device)
        self.cross_attn = CrossAttention(d_model, n_heads, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff, device=device),
            nn.GELU(),
            nn.Linear(dim_ff, d_model, device=device),
        )
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.self_attn_weights = None  # place to store the weights
        self.cross_attn_weights = None  # place to store the weights
    # end init

    def forward(self, x, mem, attn_mask=None, key_padding_mask=None):
        # Self-attention on H
        sa = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.self_attn_weights = self.self_attn.last_attn.detach()  # store for later
        x = self.norm1(x + sa)
        # Cross-attention to M
        ca = self.cross_attn(x, mem)
        self.cross_attn_weights = self.cross_attn.last_attn.detach()  # store for later
        x = self.norm2(x + ca)
        # Feedforward
        ff = self.ff(x)
        x = self.norm3(x + ff)
        return x
# end CrossTransformerBlock

# ======== Models ========

class DualEncoderModel(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, h_attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        # Melody encoder
        for layer in self.melody_encoder:
            m = layer(m)

        # Harmony encoder with cross-attn
        for layer in self.harmony_encoder:
            h = layer(h, mem=m, attn_mask=h_attn_mask)

        logits = self.out_proj(h)
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            self_attns.append(layer.self_attn_weights)
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end DualEncoderModel

class SingleEncoderModel(nn.Module):
    def __init__(
            self, 
            m_vocab_size, 
            h_vocab_size, 
            seq_len, 
            d_model=128, 
            n_heads=4, 
            num_layers=2, 
            dim_ff=256,
            device='cpu'
        ):
        super().__init__()
        self.pos = sinusoidal_positional_encoding(
            seq_len, d_model, device=device
        )
        self.seq_len = seq_len
        self.m_embed = nn.Embedding(m_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(h_vocab_size, d_model, device=device)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, device=device) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, h_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, attn_mask=None):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        x = torch.cat([m, h], dim=1)
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        logits = self.out_proj(x[:, -self.seq_len:, :])
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end SingleEncoderModel

class SimpleDE(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,
            d_model=512,
            nhead=8,
            num_layers_mel=8,
            num_layers_harm=8,
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            melody_length=80,
            harmony_length=80,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length
        
        self.pos = sinusoidal_positional_encoding(
            melody_length, d_model, device=device
        )
        self.m_embed = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.m_embed = nn.Embedding(chord_vocab_size, d_model, device=device)
        self.h_embed = nn.Embedding(chord_vocab_size, d_model, device=device)

        self.melody_encoder = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward, device=device) for _ in range(num_layers_mel)])
        self.harmony_encoder = nn.ModuleList([CrossTransformerBlock(d_model, nhead, dim_feedforward, device=device) for _ in range(num_layers_harm)])

        self.out_proj = nn.Linear(d_model, chord_vocab_size, device=device)
        self.device = device
    # end init

    def forward(self, m_seq, h_seq, h_attn_mask=None, *args, **kwargs):
        m = self.m_embed(m_seq)
        h = self.h_embed(h_seq)

        m = m + self.pos
        h = h + self.pos

        # Melody encoder
        for layer in self.melody_encoder:
            m = layer(m)

        # Harmony encoder with cross-attn
        for layer in self.harmony_encoder:
            h = layer(h, mem=m, attn_mask=h_attn_mask)

        logits = self.out_proj(h)
        return logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder:
            self_attns.append(layer.self_attn_weights)
            cross_attns.append(layer.cross_attn_weights)
        return self_attns, cross_attns
    # end get_attention_maps
# end SimpleDE
