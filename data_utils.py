import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from music21 import converter, note, chord, harmony, meter, stream
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from collections import Counter
from models_graph import remove_consecutive_duplicates, remove_out_of_dict_ids

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy

class CSGridMLMDataset(Dataset):
    def __init__(
        self,
        root_dir,
        tokenizer,
        # fixed_length=512,
        frontloading=True,
        refrontload=False,
        name_suffix='MLMH'
    ):
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') or \
                    file.endswith('.mid') or file.endswith('.midi'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.tokenizer = tokenizer
        # self.fixed_length = fixed_length
        self.frontloading = frontloading
        if self.frontloading:
            # check if file exists and load it
            root_dir = root_dir[:-1] if root_dir[-1] == '/' else root_dir
            frontloaded_file = root_dir + '_' + name_suffix + '.pickle'
            if refrontload or not os.path.isfile(frontloaded_file):
                print('Frontloading data.')
                self.encoded = []
                for data_file in tqdm(self.data_files):
                    try:
                        self.encoded.append( self.tokenizer.encode( data_file ) )
                    except Exception as e: 
                        print('Problem in:', data_file)
                        print(e)
                if frontloaded_file is not None:
                    with open(frontloaded_file, 'wb') as f:
                        pickle.dump(self.encoded, f)
            else:
                print('Loading data file.')
                with open(frontloaded_file, 'rb') as f:
                    self.encoded = pickle.load(f)
    # end init

    def __len__(self):
        if self.frontloading:
            return len(self.encoded)
        else:
            return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        if self.frontloading:
            encoded = self.encoded[idx]
        else:
            data_file = self.data_files[idx]
            encoded = self.tokenizer.encode( data_file )
        return {
            'harmony_ids': encoded['harmony_ids'],
            'attention_mask': encoded['attention_mask'],
            'pianoroll': encoded['pianoroll'],
            'time_signature': encoded['time_signature'],
            'h_density_complexity': encoded['h_density_complexity']
        }
    # end getitem
# end class dataset

def CSGridMLM_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'harmony_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
        }
    """
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    time_signature = [torch.tensor(item['time_signature'], dtype=torch.float) for item in batch]
    h_density_complexity = [torch.tensor(item['h_density_complexity'], dtype=torch.float) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]

    return {
        'harmony_ids': torch.stack(harmony_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'time_signature': torch.stack(time_signature),  # shape: (B, whatever dim)
        'h_density_complexity': torch.stack(h_density_complexity),  # shape: (B, whatever dim)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
    }
# end CSGridMLM_collate_fn

# =================== BASELINE DATASETS ==========================

class LSTMHarmonyDataset(Dataset):
    def __init__(self, base_dataset, chord_id_features):
        self.base_dataset = base_dataset
        self.chord_id_features = chord_id_features

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        chord_ids = item['harmony_ids']

        chord_ids = remove_consecutive_duplicates(chord_ids)
        chord_ids = remove_out_of_dict_ids(chord_ids, self.chord_id_features)

        return {
            "chord_sequence": torch.tensor(chord_ids, dtype=torch.long)
        }
# end LSTMHarmonyDataset

class TransitionMatrixDataset(Dataset):
    def __init__(self, base_dataset, chord_id_features, tokenizer):
        self.base_dataset = base_dataset
        self.chord_id_features = chord_id_features
        self.D = len(chord_id_features)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        chord_ids = item['harmony_ids']
        chord_ids = remove_consecutive_duplicates(chord_ids)
        chord_ids = remove_out_of_dict_ids(chord_ids, self.chord_id_features)

        matrix = torch.zeros(self.D, self.D)

        for i in range(len(chord_ids) - 1):
            a = chord_ids[i] - self.tokenizer.FIST_CHORD_TOKEN_INDEX
            b = chord_ids[i + 1] - self.tokenizer.FIST_CHORD_TOKEN_INDEX
            matrix[a, b] += 1

        # normalize rows
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = torch.where(row_sums > 0, matrix / row_sums, matrix)

        return {
            "transition_matrix": matrix
        }
# end TransitionMatrixDataset

def build_transition_vocab(dataset, chord_id_features, save_path=None):
    counter = Counter()

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        chord_ids = item['harmony_ids']

        chord_ids = remove_consecutive_duplicates(chord_ids)
        chord_ids = remove_out_of_dict_ids(chord_ids, chord_id_features)

        for j in range(len(chord_ids) - 1):
            transition = (chord_ids[j], chord_ids[j+1])
            counter[transition] += 1
    print('Making vocab')
    vocab = {t: idx for idx, t in enumerate(counter.keys())}
    print('vocab size: ', len(vocab))

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)

    return vocab
# end build_transition_vocab

def compute_CoF_PC_dist(src_feat, dst_feat):
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
    return normalized_fifths_distance + pitch_overlap
# end compute_CoF_PC_dist

class BagOfTransitionsDataset(Dataset):
    def __init__(self, base_dataset, chord_id_features, transition_vocab):
        self.base_dataset = base_dataset
        self.chord_id_features = chord_id_features
        self.transition_vocab = transition_vocab
        self.V = len(transition_vocab)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        chord_ids = item['harmony_ids']
        chord_ids = remove_consecutive_duplicates(chord_ids)
        chord_ids = remove_out_of_dict_ids(chord_ids, self.chord_id_features)

        bow = torch.zeros(self.V)

        for i in range(len(chord_ids) - 1):
            t = (chord_ids[i], chord_ids[i+1])
            if t in self.transition_vocab:
                bow[self.transition_vocab[t]] += 1
            else:
                idx = self.find_most_similar_transition(t)
                bow[idx] += 1

        # normalization
        if bow.sum() > 0:
            bow = bow / bow.sum()

        return {
            "bag_of_transitions": bow
        }
    def find_most_similar_transition(self, trans):
        similar_idx = 0
        shortest_distance = float('inf')
        for i, t in enumerate(self.transition_vocab):
            vocab_1 = torch.tensor(self.chord_id_features[t[0]])
            vocab_2 = torch.tensor(self.chord_id_features[t[1]])
            new_1 = torch.tensor(self.chord_id_features[trans[0]])
            new_2 = torch.tensor(self.chord_id_features[trans[1]])
            tmp_dist = compute_CoF_PC_dist( vocab_1, new_1 )
            tmp_dist += compute_CoF_PC_dist( vocab_2, new_2 )
            if shortest_distance > tmp_dist:
                shortest_distance = tmp_dist
                similar_idx = i
        return similar_idx
    # end find_most_similar_transition
# end BagOfTransitionsDataset

class LSTMPaddingCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        # batch: list of dicts with "chord_sequence"

        sequences = [item["chord_sequence"] for item in batch]

        # Convert to tensors if not already
        # sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

        # Pad
        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.pad_id
        )

        # Build input and target (shifted)
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]

        # Attention mask (1 where not pad)
        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask
        }
# end LSTMPaddingCollator

def compute_masked_loss(logits, targets, attention_mask):
    # logits: (B, T, V)
    # targets: (B, T)
    # mask: (B, T)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    )

    loss = loss.view(targets.size())

    # Apply mask
    loss = loss * attention_mask

    return loss.sum() / attention_mask.sum()
# end compute_masked_loss