import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# -----------------------------
# GCN-based Graph Encoder
# -----------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        pooled = global_mean_pool(x, batch)
        mu = self.mu_proj(pooled)
        logvar = self.logvar_proj(pooled)
        return mu, logvar

# -----------------------------
# Decoder: from z → node features
# -----------------------------
class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, z, num_nodes):
        z_expanded = z.unsqueeze(1).repeat(1, num_nodes, 1).view(-1, z.size(1))
        h = F.relu(self.fc1(z_expanded))
        out = torch.sigmoid(self.fc2(h))  # 24D output per node
        return out

# -----------------------------
# Full VAE
# -----------------------------
class GraphVAE(nn.Module):
    def __init__(self, in_dim=24, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, in_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, data.num_nodes // data.num_graphs)
        return recon_x, mu, logvar

# -----------------------------
# VAE Loss
# -----------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

# -----------------------------
# Helper: turn chord sequence to PyG graph
# -----------------------------
def make_graph_from_chords(chord_ids_sequence, chord_id_features):
    unique_ids = list(set(chord_ids_sequence))
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

    x = torch.stack([torch.LongTensor(chord_id_features[cid]) for cid in unique_ids])  # [num_unique_nodes, 24]

    edges = []
    for i in range(len(chord_ids_sequence) - 1):
        src = id_to_idx[chord_ids_sequence[i]]
        dst = id_to_idx[chord_ids_sequence[i + 1]]
        edges.append((src, dst))

    edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, num_edges]
    
    return Data(x=x, edge_index=edge_index)
# end make_graph_from_chords

def remove_consecutive_duplicates(lst):
    if not lst:
        return []
    result = [lst[0]]
    for item in lst[1:]:
        if item != result[-1]:
            result.append(item)
    return result
# end remove_consecutive_duplicates

def remove_out_of_dict_ids(lst, d):
    if not lst:
        return []
    result = []
    for item in lst:
        if item in d.keys():
            result.append( item )
    return result
# end remove_out_of_dict_ids

def make_graph_from_input_ids(chord_id_duplicates_sequence, chord_id_features):
    chord_ids_sequence = remove_consecutive_duplicates( chord_id_duplicates_sequence )
    chord_ids_sequence = remove_out_of_dict_ids( chord_ids_sequence, chord_id_features )
    return make_graph_from_chords(chord_ids_sequence, chord_id_features)
# end make_graph_from_input_ids

# -----------------------------
# Usage
# -----------------------------
# Assuming:
# - `chord_id_features`: dict {int: torch.FloatTensor([24])}
# - `chord_ids_sequence`: list of chord ids like [5, 12, 7, ...]

# Example:
# graph = make_graph_from_chords(chord_ids_sequence, chord_id_features)
# batch = Batch.from_data_list([graph1, graph2, ...])
# model = GraphVAE()
# recon_x, mu, logvar = model(batch)
# loss = vae_loss(recon_x, batch.x, mu, logvar)

# -----------------------------
# transformer
# -----------------------------

class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, use_cross_attention=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Optional cross-attention components
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.norm_cross = nn.LayerNorm(d_model)
    # end init

    def forward(self, x, guide=None, guide_mask=None):
        # Self-attention
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)

        # Optional cross-attention
        if self.use_cross_attention and guide is not None:
            ca_out, _ = self.cross_attn(x, guide, guide, key_padding_mask=guide_mask)
            x = self.norm_cross(x + ca_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
    # end forward
# end CrossAttentionEncoderLayer
