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

lstm_train_dataset = LSTMHarmonyDataset(train_dataset, chord_id_features)
lstm_val_dataset = LSTMHarmonyDataset(val_dataset, chord_id_features)

collator = LSTMPaddingCollator(pad_id=0)

train_loader_lstm = DataLoader(
    lstm_train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator
)
val_loader_lstm = DataLoader(
    lstm_val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collator
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_lstm = LSTMHarmonyModel(vocab_size=len(chord_id_features) + tokenizer.FIST_CHORD_TOKEN_INDEX)
model_lstm.train()
model_lstm.to(device)

optimizer_lstm = torch.optim.AdamW(model_lstm.parameters(), lr=1e-4)

os.makedirs('saved_models', exist_ok=True)

train_lstm(model_lstm, train_loader_lstm, val_loader_lstm, optimizer_lstm, device, save_path='saved_models/lstm.pt', num_epochs=200)