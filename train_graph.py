import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from models_graph import make_graph_from_input_ids, remove_consecutive_duplicates, remove_out_of_dict_ids, compute_edge_features
from data_utils import CSGridMLMDataset
import torch
from models import HarmonicGraphEncoder, BilinearDecoder, HarmonicGAE, edge_index_to_dense

tokenizer = CSGridMLMTokenizer(fixed_length=256)
chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

train_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'

train_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')