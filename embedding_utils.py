from data_utils import LSTMHarmonyDataset, TransitionMatrixDataset, BagOfTransitionsDataset, LSTMPaddingCollator
import torch
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import pickle
from models_baseline import LSTMHarmonyModel, TransitionMatrixAutoencoder, BagOfTransitionsAutoencoder
from models_graph import HarmonicGAE, make_graph_from_input_ids
from models import SEFiLMModel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

BoT_vocab_path = 'data/BoT_vocab.pickle'

with open(BoT_vocab_path, 'rb') as f:
    BoT_vocab = pickle.load(f)

device_name = 'cuda:0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_lstm_embeddings_for_dataset(loaded_dataset, tokenizer):

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    lstm_val_dataset = LSTMHarmonyDataset(loaded_dataset, chord_id_features)
    model = LSTMHarmonyModel(vocab_size=len(chord_id_features)+tokenizer.FIST_CHORD_TOKEN_INDEX)
    checkpoint = torch.load('saved_models/lstm.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    lstm_embeddings = []
    for d in lstm_val_dataset:
        tmp_seq = d['chord_sequence'].to(device)
        if len(tmp_seq) > 1:
            model_out = model(tmp_seq)
            lstm_embeddings.append( model_out[1][0][1,:].detach().cpu() )
    lstm_np = np.array(lstm_embeddings)
    return lstm_np
# end get_lstm_embeddings_for_data_path

def get_matrix_embeddings_for_dataset(loaded_dataset, tokenizer):

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    matrix_val_dataset = TransitionMatrixDataset(loaded_dataset, chord_id_features, tokenizer)

    model = TransitionMatrixAutoencoder(D=len(chord_id_features))
    checkpoint = torch.load('saved_models/matrix.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    matrix_embeddings = []
    for d in matrix_val_dataset:
        tmp_matrix = d['transition_matrix'].to(device).unsqueeze(0)
        if tmp_matrix.sum() > 0:
            model_out = model(tmp_matrix)
            matrix_embeddings.append( model_out[1][0].detach().cpu() )
    matrix_np = np.array(matrix_embeddings)
    return matrix_np
# end get_matrix_embeddings_for_data_path

def get_bot_embeddings_for_dataset(loaded_dataset, tokenizer):

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    bot_val_dataset = BagOfTransitionsDataset(loaded_dataset, chord_id_features, BoT_vocab)

    model = BagOfTransitionsAutoencoder(vocab_size=len(BoT_vocab))
    checkpoint = torch.load('saved_models/bot.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    bot_embeddings = []
    for d in bot_val_dataset:
        tmp_bot = d['bag_of_transitions'].to(device).unsqueeze(0)
        if tmp_bot.sum() > 0:
            model_out = model(tmp_bot)
            bot_embeddings.append( model_out[1][0].detach().cpu() )
    bot_np = np.array(bot_embeddings)
    return bot_np
# end get_bot_embeddings_for_data_path

def get_graph_embeddings_for_dataset(loaded_dataset, tokenizer):

    chord_features = GridMLM_tokenizers.CHORD_FEATURES
    chord_id_features = {tokenizer.vocab[k]: v for k, v in chord_features.items()}

    graph_val_dataset = []

    for d in tqdm(loaded_dataset):
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

    model = HarmonicGAE(hidden_dim=64)
    checkpoint = torch.load('saved_models/gae.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()

    graph_embeddings = []

    for d in graph_val_dataset:
        _, emb = model.encoder(d)
        graph_embeddings.append(emb.detach().numpy().squeeze())
    
    graph_np = np.array(graph_embeddings)
    return graph_np
# end get_graph_embeddings_for_data_path

def get_SE_embeddings_for_dataset(loaded_dataset, tokenizer):
    model = SEFiLMModel(
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
    model.load_state_dict(checkpoint)
    model.eval()

    transformer_embeddings = []

    for d in loaded_dataset:
        melody_grid = torch.FloatTensor( d['pianoroll'] ).reshape( 1, d['pianoroll'].shape[0], d['pianoroll'].shape[1] )
        harmony_real = torch.LongTensor(d['harmony_ids']).reshape(1, len(d['harmony_ids']))
        _, hidden = model(
            melody_grid=melody_grid.to(model.device),
            harmony_tokens=harmony_real.to(model.device),
            guidance_embedding=None,
            return_hidden=True
        )
        transformer_embeddings.append(hidden.detach().cpu().numpy().squeeze())
    
    transformer_np = np.array(transformer_embeddings)
    return transformer_np
# end get_SE_embeddings_for_data_path