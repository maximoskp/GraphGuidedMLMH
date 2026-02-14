from data_utils import LSTMHarmonyDataset, TransitionMatrixDataset, BagOfTransitionsDataset, LSTMPaddingCollator
from torch.utils.data import DataLoader
import torch
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import pickle
from models_baseline import LSTMHarmonyModel, TransitionMatrixAutoencoder, BagOfTransitionsAutoencoder
from models_baseline import train_lstm, train_matrix_ae, train_bot_ae
import os

# train_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_train'
# val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'
train_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'

tokenizer = CSGridMLMTokenizer(fixed_length=256)

train_dataset = CSGridMLMDataset(train_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

matrix_train_dataset = TransitionMatrixDataset(train_dataset, chord_id_features, tokenizer)
matrix_val_dataset = TransitionMatrixDataset(val_dataset, chord_id_features, tokenizer)

train_loader_matrix = DataLoader(
    matrix_train_dataset,
    batch_size=32,
    shuffle=True
)
val_loader_matrix = DataLoader(
    matrix_val_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_matrix = TransitionMatrixAutoencoder(D=len(chord_id_features))
model_matrix.train()
model_matrix.to(device)

optimizer_matrix = torch.optim.AdamW(model_matrix.parameters(), lr=1e-4)

os.makedirs('saved_models', exist_ok=True)

train_matrix_ae(model_matrix, train_loader_matrix, val_loader_matrix, optimizer_matrix, device, save_path='saved_models/matrix.pt', num_epochs=200)