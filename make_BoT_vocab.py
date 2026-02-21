import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from utils.data_utils import CSGridMLMDataset, build_transition_vocab
import os

train_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_train'

tokenizer = CSGridMLMTokenizer(fixed_length=256)

train_dataset = CSGridMLMDataset(train_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

chord_features = GridMLM_tokenizers.CHORD_FEATURES

chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

os.makedirs('data', exist_ok=True)

build_transition_vocab(train_dataset, chord_id_features, save_path='data/BoT_vocab.pickle')