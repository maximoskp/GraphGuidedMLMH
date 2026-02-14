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

BoT_vocab_path = 'data/BoT_vocab.pickle'

with open(BoT_vocab_path, 'rb') as f:
    BoT_vocab = pickle.load(f)

# train_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_train'
# val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'
train_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'

tokenizer = CSGridMLMTokenizer(fixed_length=256)

train_dataset = CSGridMLMDataset(train_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

bot_train_dataset = BagOfTransitionsDataset(train_dataset, chord_id_features, BoT_vocab)
bot_val_dataset = BagOfTransitionsDataset(val_dataset, chord_id_features, BoT_vocab)

train_loader_bot = DataLoader(
    bot_train_dataset,
    batch_size=32,
    shuffle=True
)
val_loader_bot = DataLoader(
    bot_val_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_bot = BagOfTransitionsAutoencoder(vocab_size=len(BoT_vocab))
model_bot.train()
model_bot.to(device)

optimizer_bot = torch.optim.AdamW(model_bot.parameters(), lr=1e-4)

os.makedirs('saved_models', exist_ok=True)

train_bot_ae(model_bot, train_loader_bot, val_loader_bot, optimizer_bot, device, save_path='saved_models/bot.pt', num_epochs=200)