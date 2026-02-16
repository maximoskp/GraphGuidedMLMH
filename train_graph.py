import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from models_graph import make_graph_from_input_ids, remove_consecutive_duplicates, remove_out_of_dict_ids, compute_edge_features
from data_utils import CSGridMLMDataset
import torch
from models_graph import HarmonicGAE, train_gae
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import os

tokenizer = CSGridMLMTokenizer(fixed_length=256)
chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

train_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'

train_dataset = CSGridMLMDataset(train_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

# create graph datasets as lists of graph data objects
print('Making graph training dataset.')
graph_train_dataset = []
for d in tqdm(train_dataset):
    chord_id_duplicates_sequence = d['harmony_ids']
    g = make_graph_from_input_ids(
            chord_id_duplicates_sequence,
            chord_id_features,
            use_probabilities=True
        )
    if g is not None:
        graph_train_dataset.append( g )
    else:
        print('Short sequence: ', chord_id_duplicates_sequence)

print('Making graph validation dataset.')
graph_val_dataset = []
for d in tqdm(val_dataset):
    chord_id_duplicates_sequence = d['harmony_ids']
    g = make_graph_from_input_ids(
            chord_id_duplicates_sequence,
            chord_id_features,
            use_probabilities=True
        )
    if g is not None:
        graph_val_dataset.append( g )
    else:
        print('Short sequence: ', chord_id_duplicates_sequence)

print('Making data loaders.')
trainloader = DataLoader(graph_train_dataset, batch_size=32, shuffle=False)
valloader = DataLoader(graph_val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hidden_dim = 64

model = HarmonicGAE(hidden_dim=hidden_dim)

optimizer_gae = torch.optim.AdamW(model.parameters(), lr=1e-5)

os.makedirs('saved_models', exist_ok=True)

train_gae(model, trainloader, valloader, optimizer_gae, device, save_path='saved_models/gae.pt', num_epochs=200)