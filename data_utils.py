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

# def extract_lead_sheet_info(xml_path, quantization='16th', fixed_length=None):
#     # Load the score and flatten
#     score = converter.parse(xml_path)
#     melody_part = score.parts[0].flat

#     # Define 16th note length
#     ql_per_16th = 0.25 / 4

#     # Get pitch range: MIDI 21 (A0) to 108 (C8) -> 88 notes
#     pitch_range = list(range(21, 109))
#     n_pitch = len(pitch_range)

#     # Step 1: Find first chord symbol and bar to trim before it
#     first_chord_offset = None
#     for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
#         first_chord_offset = el.offset
#         break

#     measure_start_offset = 0.0
#     if first_chord_offset is not None:
#         for meas in melody_part.getElementsByClass(stream.Measure):
#             if meas.offset <= first_chord_offset < meas.offset + meas.duration.quarterLength:
#                 measure_start_offset = meas.offset
#                 break

#     skip_steps = int(np.round(measure_start_offset / ql_per_16th))

#     # Determine total length in 16th notes
#     total_duration_q = melody_part.highestTime
#     total_steps = int(np.ceil(total_duration_q / ql_per_16th))

#     # Allocate raw matrices (we will trim/pad later)
#     raw_pianoroll = np.zeros((total_steps, n_pitch), dtype=np.uint8)
#     raw_chords = [None] * total_steps

#     # Fill pianoroll
#     for el in melody_part.notesAndRests:
#         start = int(np.round(el.offset / ql_per_16th))
#         dur_steps = int(np.round(el.quarterLength / ql_per_16th))

#         if isinstance(el, note.Note):
#             midi = el.pitch.midi
#             if midi in pitch_range:
#                 idx = pitch_range.index(midi)
#                 raw_pianoroll[start:start+dur_steps, idx] = 1

#         elif isinstance(el, chord.Chord):  # Just in case
#             for pitch in el.pitches:
#                 midi = pitch.midi
#                 if midi in pitch_range:
#                     idx = pitch_range.index(midi)
#                     raw_pianoroll[start:start+dur_steps, idx] = 1

#     # Fill chord grid
#     for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
#         start = int(np.round(el.offset / ql_per_16th))
#         if 0 <= start < len(raw_chords):
#             raw_chords[start] = el.figure

#     # Propagate chord forward
#     for i in range(1, len(raw_chords)):
#         if raw_chords[i] is None:
#             raw_chords[i] = raw_chords[i-1]

#     # Fill missing with <pad> or <nc>
#     for i in range(len(raw_chords)):
#         if raw_chords[i] is None:
#             raw_chords[i] = "<nc>"  # Or use "<pad>" if preferred

#     # Trim to start at first chord bar
#     raw_pianoroll = raw_pianoroll[skip_steps:]
#     raw_chords = raw_chords[skip_steps:]

#     # Add pitch class profile (top 12 dims)
#     n_steps = len(raw_pianoroll)
#     pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
#     for i in range(n_steps):
#         pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
#         for idx in pitch_indices:
#             midi = pitch_range[idx]
#             pitch_classes[i, midi % 12] = 1
#     full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])  # Shape: (T, 12 + 88)

#     # Apply fixed length (pad or trim)
#     if fixed_length is not None:
#         if n_steps > fixed_length:
#             full_pianoroll = full_pianoroll[:fixed_length]
#             raw_chords = raw_chords[:fixed_length]
#         elif n_steps < fixed_length:
#             pad_len = fixed_length - n_steps
#             pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
#             pad_ch = ["<pad>"] * pad_len
#             full_pianoroll = np.vstack([full_pianoroll, pad_pr])
#             raw_chords += pad_ch

#     return full_pianoroll, raw_chords
# #  end extract_lead_sheet_info

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
    def __init__(self, base_dataset, chord_id_features):
        self.base_dataset = base_dataset
        self.chord_id_features = chord_id_features
        self.D = len(chord_id_features)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        chord_ids = item['harmony_ids']
        chord_ids = remove_consecutive_duplicates(chord_ids)
        chord_ids = remove_out_of_dict_ids(chord_ids, self.chord_id_features)

        matrix = torch.zeros(self.D, self.D)

        for i in range(len(chord_ids) - 1):
            a = chord_ids[i]
            b = chord_ids[i + 1]
            matrix[a, b] += 1

        # Optional: normalize rows
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

        # Optional normalization
        if bow.sum() > 0:
            bow = bow / bow.sum()

        return {
            "bag_of_transitions": bow
        }
# end BagOfTransitionsDataset

class LSTMPaddingCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        # batch: list of dicts with "chord_sequence"

        sequences = [item["chord_sequence"] for item in batch]

        # Convert to tensors if not already
        sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

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