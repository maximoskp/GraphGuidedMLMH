from data_utils import LSTMHarmonyDataset, TransitionMatrixDataset, BagOfTransitionsDataset, LSTMPaddingCollator
import torch
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import pickle
from models_baseline import LSTMHarmonyModel, TransitionMatrixAutoencoder, BagOfTransitionsAutoencoder
from models_graph import HarmonicGAE, make_graph_from_input_ids, remove_consecutive_duplicates, remove_out_of_dict_ids
from models import SEFiLMModel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

BoT_vocab_path = 'data/BoT_vocab.pickle'

with open(BoT_vocab_path, 'rb') as f:
    BoT_vocab = pickle.load(f)

device_name = 'cuda:0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

chord_features = GridMLM_tokenizers.CHORD_FEATURES
chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}
D = len(chord_id_features)
V = len(BoT_vocab)

# load models ====================================================================================
model_lstm = LSTMHarmonyModel(vocab_size=len(chord_id_features)+tokenizer.FIST_CHORD_TOKEN_INDEX)
checkpoint = torch.load('saved_models/lstm.pt', map_location=device_name)
model_lstm.load_state_dict(checkpoint)
model_lstm.eval()
# model_lstm.to(device)

model_matrix = TransitionMatrixAutoencoder(D=len(chord_id_features))
checkpoint = torch.load('saved_models/matrix.pt', map_location=device_name)
model_matrix.load_state_dict(checkpoint)
model_matrix.eval()
# model_matrix.to(device)

model_bot = BagOfTransitionsAutoencoder(vocab_size=len(BoT_vocab))
checkpoint = torch.load('saved_models/bot.pt', map_location=device_name)
model_bot.load_state_dict(checkpoint)
model_bot.eval()
# model_bot.to(device)

model_graph = HarmonicGAE(hidden_dim=64)
checkpoint = torch.load('saved_models/gae.pt', map_location=device_name)
model_graph.load_state_dict(checkpoint)
model_graph.eval()

model_SE = SEFiLMModel(
    chord_vocab_size=len(tokenizer.vocab),
    d_model=512,
    nhead=8,
    num_layers=8,
    grid_length=80,
    pianoroll_dim=tokenizer.pianoroll_dim,
    guidance_dim=None,
    device=device,
)
checkpoint = torch.load('saved_models/SE/pretrained.pt', map_location=device_name)
model_SE.load_state_dict(checkpoint)
model_SE.eval()

# get embeddings functions ===========================================================
def get_lstm_embeddings_for_sequence(seq):
    model_out = model_lstm(seq)
    return model_out[1][0][1,:].detach().cpu()
# end lstm
def get_matrix_embeddings_for_sequence(mat):
    model_out = model_matrix(mat)
    return model_out[1][0].detach().cpu()
# end matrix
def get_bot_embeddings_for_sequence(mat):
    model_out = model_bot(mat)
    return model_out[1][0].detach().cpu()
# end bot
def get_SE_embeddings_for_sequence(pianoroll, harmony_ids):
    melody_grid = torch.FloatTensor( pianoroll ).reshape( 1, pianoroll.shape[0], pianoroll.shape[1] )
    harmony_real = torch.LongTensor(harmony_ids).reshape(1, len(harmony_ids))
    _, hidden = model_SE(
        melody_grid=melody_grid.to(model_SE.device),
        harmony_tokens=harmony_real.to(model_SE.device),
        guidance_embedding=None,
        return_hidden=True
    )
    return hidden.detach().cpu().squeeze()
# end SE

def make_contrastive_bundle_dataset_for_dataset(dataset):

    bundled_dataset = []

    # make temporary BoT dataset for enabling useful functions
    bot_dataset = BagOfTransitionsDataset(dataset, chord_id_features, BoT_vocab)

    for d in tqdm(dataset):
        new_data = {}
        new_data['harmony_ids'] = d['harmony_ids']
        new_data['pianoroll'] = d['pianoroll']
        seq = d['harmony_ids']
        # lstm
        seq = remove_consecutive_duplicates(seq)
        seq = remove_out_of_dict_ids(seq, chord_id_features)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if len(seq) > 1:
            # lstm
            lstm_embeddings = get_lstm_embeddings_for_sequence(seq_tensor)
            new_data['lstm_embeddings'] = lstm_embeddings
            # matrix
            matrix = torch.zeros(D, D)
            for i in range(len(seq) - 1):
                a = seq[i] - tokenizer.FIST_CHORD_TOKEN_INDEX
                b = seq[i + 1] - tokenizer.FIST_CHORD_TOKEN_INDEX
                matrix[a, b] += 1
            # normalize rows
            row_sums = matrix.sum(dim=1, keepdim=True)
            matrix = torch.where(row_sums > 0, matrix / row_sums, matrix).unsqueeze(0)
            new_data['matrix'] = matrix
            matrix_embeddings = get_matrix_embeddings_for_sequence(matrix)
            new_data['matrix_embeddings'] = matrix_embeddings
            # BoT
            bow = torch.zeros(V)
            for i in range(len(seq) - 1):
                t = (seq[i], seq[i+1])
                if t in BoT_vocab:
                    bow[BoT_vocab[t]] += 1
                else:
                    idx = bot_dataset.find_most_similar_transition(t)
                    bow[idx] += 1
            # normalization
            if bow.sum() > 0:
                bow = bow / bow.sum()
            new_data['bot'] = bow
            bot_embeddings = get_bot_embeddings_for_sequence(bow)
            new_data['bot_embeddings'] = bot_embeddings
            # graph
            g = make_graph_from_input_ids(
                d['harmony_ids'],
                chord_id_features,
                use_probabilities=True
            )
            _, emb = model_graph.encoder(g)
            new_data['graph'] = g
            new_data['graph_embeddings'] = emb.detach().squeeze()
            # SE
            hidden = get_SE_embeddings_for_sequence(new_data['pianoroll'], d['harmony_ids'])
            new_data['transformer_embeddings'] = hidden

            bundled_dataset.append(new_data)
        # end if
    # end for
    return bundled_dataset
# end make_contrastive_bundle_dataset_for_dataset

def main():
    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    parent_main_dirs = '/mnt/ssd2/maximos/data/hooktheory_midi_hr'
    dataset_subdirs = ['CA_train', 'CA_test']
    parent_dir_idioms = '/mnt/ssd2/maximos/data/coinvent_midi'
    other_dirs = os.listdir(parent_dir_idioms)
    idiom_dirs = [f for f in other_dirs if '.pickle' not in f]

    full_dirs = []
    names = []
    for d in dataset_subdirs:
        full_dirs.append(os.path.join(parent_main_dirs, d))
        names.append(d)
    for d in idiom_dirs:
        full_dirs.append(os.path.join(parent_dir_idioms, d))
        names.append(d)

    for d, n in zip(full_dirs, names):
        print(f'running for {n} in path: {d}')
        train_dataset = CSGridMLMDataset(d, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
        bundled = make_contrastive_bundle_dataset_for_dataset(train_dataset)
        print('saving...')
        with open(f'data/{n}.pickle', 'wb') as f:
            pickle.dump(bundled, f)
# end main

if __name__ == '__main__':
    main()