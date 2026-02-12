import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, Sequential, Linear, ReLU
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
from collections import defaultdict
import math
from copy import deepcopy

### ======================= GRAPH-GUIDED MODELS =======================
# -----------------------------
# Helper: turn chord sequence to PyG graph
# -----------------------------

def compute_edge_features(src_feat, dst_feat):
    """
    src_feat, dst_feat: torch tensors of shape [24]
    
    Returns:
        torch.tensor([normalized_fifths_distance, pitch_class_overlap])
    """
    
    # ---- Root extraction ----
    src_root = torch.argmax(src_feat[:12]).item()
    dst_root = torch.argmax(dst_feat[:12]).item()
    
    # Map to circle of fifths index
    src_circle = (src_root * 7) % 12
    dst_circle = (dst_root * 7) % 12
    
    # Circular distance
    diff = abs(src_circle - dst_circle)
    fifths_distance = min(diff, 12 - diff)
    
    # Normalize to [0,1]
    normalized_fifths_distance = fifths_distance / 6.0
    
    # ---- Pitch-class overlap (Jaccard similarity) ----
    src_pcs = src_feat[12:].bool()
    dst_pcs = dst_feat[12:].bool()
    
    intersection = torch.logical_and(src_pcs, dst_pcs).sum().item()
    union = torch.logical_or(src_pcs, dst_pcs).sum().item()
    
    if union == 0:
        pitch_overlap = 0.0
    else:
        pitch_overlap = intersection / union
    
    return torch.tensor([normalized_fifths_distance, pitch_overlap], dtype=torch.float)
# end compute_edge_features

# def make_graph_from_chords(chord_ids_sequence, chord_id_features):
#     unique_ids = list(set(chord_ids_sequence))
#     id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

#     x = torch.stack([torch.LongTensor(chord_id_features[cid]) for cid in unique_ids])  # [num_unique_nodes, 24]

#     edges = []
#     for i in range(len(chord_ids_sequence) - 1):
#         src = id_to_idx[chord_ids_sequence[i]]
#         dst = id_to_idx[chord_ids_sequence[i + 1]]
#         edges.append((src, dst))

#     edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, num_edges]
    
#     return Data(x=x, edge_index=edge_index)
# # end make_graph_from_chords

def make_weighted_graph_from_chords(chord_ids_sequence, chord_id_features, use_probabilities=False):
    
    unique_ids = list(set(chord_ids_sequence))
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

    # ---- Node feature matrix ----
    x = torch.stack([
        torch.tensor(chord_id_features[cid], dtype=torch.float)
        for cid in unique_ids
    ])

    # ---- Count transitions ----
    transition_counts = defaultdict(int)
    outgoing_counts = defaultdict(int)

    for i in range(len(chord_ids_sequence) - 1):
        src_id = chord_ids_sequence[i]
        dst_id = chord_ids_sequence[i + 1]

        if src_id in id_to_idx and dst_id in id_to_idx:
            transition_counts[(src_id, dst_id)] += 1
            outgoing_counts[src_id] += 1

    # ---- Build edges ----
    edge_index = []
    edge_weight = []
    edge_attr = []

    for (src_id, dst_id), count in transition_counts.items():
        
        src_idx = id_to_idx[src_id]
        dst_idx = id_to_idx[dst_id]

        edge_index.append([src_idx, dst_idx])

        # weight = count or probability
        if use_probabilities:
            weight = count / outgoing_counts[src_id]
        else:
            weight = float(count)

        edge_weight.append(weight)

        # compute edge features
        src_feat = x[src_idx]
        dst_feat = x[dst_idx]
        ef = compute_edge_features(src_feat, dst_feat)
        edge_attr.append(ef)

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # [2, num_edges]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_attr = torch.stack(edge_attr)  # [num_edges, 2]

    return Data(
        x=x,
        node_ids={v:k for k,v in id_to_idx.items()},  # mapping back to chord ids
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_attr
    )
# end make_weighted_graph_from_chords

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

def make_graph_from_input_ids(chord_id_duplicates_sequence, chord_id_features, use_probabilities=True):
    chord_ids_sequence = remove_consecutive_duplicates( chord_id_duplicates_sequence )
    chord_ids_sequence = remove_out_of_dict_ids( chord_ids_sequence, chord_id_features )
    return make_weighted_graph_from_chords(chord_ids_sequence, chord_id_features, use_probabilities=use_probabilities)
    # return make_graph_from_chords(chord_ids_sequence, chord_id_features)
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


# ================ GRAPH models ====================

class HarmonicGraphEncoder(torch.nn.Module):
    def __init__(self, node_dim=24, edge_dim=2, hidden_dim=64):
        super().__init__()

        # Node projection
        self.lin_in = Linear(node_dim, hidden_dim)

        # Edge projection (CRUCIAL)
        self.edge_proj = Linear(edge_dim, hidden_dim)

        # Update MLP used by GINE
        nn_update = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(nn_update)
        self.conv2 = GINEConv(nn_update)
    # end init

    def forward(self, data):
        x = self.lin_in(data.x)

        # project edges to same dim as nodes
        edge_attr = self.edge_proj(data.edge_attr)

        x = self.conv1(x, data.edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr)

        return x
    # end forward
# end HarmonicGraphEncoder

class BilinearDecoder(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.M = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))
    # end init

    def forward(self, node_emb):
        # node_emb: [N, H]
        logits = node_emb @ self.M @ node_emb.T
        probs = torch.softmax(logits, dim=1)
        return probs
    # end forward
# end BilinearDecoder

class HarmonicGAE(torch.nn.Module):
    def __init__(self, node_dim=24, edge_dim=2, hidden_dim=64):
        super().__init__()
        self.encoder = HarmonicGraphEncoder(node_dim, edge_dim, hidden_dim)
        self.decoder = BilinearDecoder(hidden_dim)
    # end init

    def forward(self, data):
        node_emb = self.encoder(data)
        probs = self.decoder(node_emb)
        return probs
    # end forward
# end HarmonicGAE

def edge_index_to_dense(data):
    N = data.x.size(0)
    W = torch.zeros((N, N), device=data.x.device)

    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i]
        dst = data.edge_index[1, i]
        W[src, dst] = data.edge_weight[i]

    # return torch.softmax(W, dim=1)
    return W
# end edge_index_to_dense

'''
pred = model(data)
target = edge_index_to_dense(data)

loss = F.mse_loss(pred, target)
'''