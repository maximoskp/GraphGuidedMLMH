from data_utils import LSTMHarmonyDataset, TransitionMatrixDataset, BagOfTransitionsDataset, LSTMPaddingCollator
import torch
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import pickle
from models_baseline import LSTMHarmonyModel, TransitionMatrixAutoencoder, BagOfTransitionsAutoencoder
from models_graph import HarmonicGAE, make_graph_from_input_ids
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

BoT_vocab_path = 'data/BoT_vocab.pickle'

with open(BoT_vocab_path, 'rb') as f:
    BoT_vocab = pickle.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_lstm_embeddings_for_data_path(data_path):
    tokenizer = CSGridMLMTokenizer(fixed_length=256)

    val_dataset = CSGridMLMDataset(data_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    lstm_val_dataset = LSTMHarmonyDataset(val_dataset, chord_id_features)
    model_lstm = LSTMHarmonyModel(vocab_size=len(chord_id_features)+tokenizer.FIST_CHORD_TOKEN_INDEX)
    model_lstm.eval()
    model_lstm.to(device)

    lstm_embeddings = []
    for d in lstm_val_dataset:
        model_out = model_lstm(d['chord_sequence'].to(device))
        lstm_embeddings.append( model_out[1][0][1,:].detach().cpu() )
    lstm_np = np.array(lstm_embeddings)
    return lstm_np
# end get_lstm_embeddings_for_data_path

def get_matrix_embeddings_for_data_path(data_path):
    tokenizer = CSGridMLMTokenizer(fixed_length=256)

    val_dataset = CSGridMLMDataset(data_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    matrix_val_dataset = TransitionMatrixDataset(val_dataset, chord_id_features, tokenizer)

    model_matrix = TransitionMatrixAutoencoder(D=len(chord_id_features))
    model_matrix.train()
    model_matrix.to(device)

    matrix_embeddings = []
    for d in matrix_val_dataset:
        model_out = model_matrix(d['transition_matrix'].to(device).unsqueeze(0))
        matrix_embeddings.append( model_out[1][0].detach().cpu() )
    matrix_np = np.array(matrix_embeddings)
    return matrix_np
# end get_matrix_embeddings_for_data_path

def get_bot_embeddings_for_data_path(data_path):
    tokenizer = CSGridMLMTokenizer(fixed_length=256)

    val_dataset = CSGridMLMDataset(data_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    bot_val_dataset = BagOfTransitionsDataset(val_dataset, chord_id_features, BoT_vocab)

    model_bot = BagOfTransitionsAutoencoder(vocab_size=len(BoT_vocab))
    model_bot.train()
    model_bot.to(device)

    bot_embeddings = []
    for d in bot_val_dataset:
        model_out = model_bot(d['bag_of_transitions'].to(device).unsqueeze(0))
        bot_embeddings.append( model_out[1][0].detach().cpu() )
    bot_np = np.array(bot_embeddings)
    return bot_np
# end get_bot_embeddings_for_data_path

def get_graph_embeddings_for_data_path(data_path):
    tokenizer = CSGridMLMTokenizer(fixed_length=256)

    val_dataset = CSGridMLMDataset(data_path, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

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

    model_graph = HarmonicGAE(hidden_dim=64)
    model_graph.train()

    graph_embeddings = []

    for d in graph_val_dataset:
        _, emb = model_graph.encoder(d)
        graph_embeddings.append(emb.detach().numpy)
    
    graph_np = np.array(graph_embeddings)
    return graph_np
# end get_graph_embeddings_for_data_path