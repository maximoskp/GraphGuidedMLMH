from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from models import ContrastiveSpaceModel
import pickle
import numpy as np
import os
from train_utils import train_contrastive
import argparse

spaces = {'lstm', 'matrix', 'bot', 'graph'}

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-s', '--source_name', type=str, help='Specify the sources space name among: ' + repr(spaces), required=True)
    parser.add_argument('-d', '--shared_dim', type=int, help='Specify the shared dimension (defaults in 64).', required=False)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 1e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 32.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    if args.source_name:
        source_name = args.source_name
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    shared_dim = 64
    if args.shared_dim:
        shared_dim = args.shared_dim
    epochs = 100
    if args.epochs:
        epochs = args.epochs
    lr = 1e-5
    if args.learningrate:
        lr = args.learningrate
    batch_size = 32
    if args.batchsize:
        batch_size = args.batchsize

    source_key = source_name + '_embeddings'

    train_path = 'data/contrastive_dataset/CA_train.pickle'
    val_path = 'data/contrastive_dataset/CA_test.pickle'

    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)

    source_dim = train_dataset[0][source_key].shape[0]
    transformer_dim = train_dataset[0]['transformer_embeddings'].shape[0]
    print(source_name, ' - source_dim: ', source_dim)
    print('transformer', ' - transformer_dim: ', transformer_dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    model = ContrastiveSpaceModel(source_dim, transformer_dim, shared_dim=shared_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    results_path = os.path.join( 'results', 'contrastive', f'{source_name}.csv' )
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/contrastive', exist_ok=True)

    save_path = os.path.join( 'saved_models', 'contrastive', f'{source_name}.csv' )
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_models/contrastive', exist_ok=True)

    train_contrastive(
            model,
            train_loader,
            val_loader,
            optimizer,
            results_path,
            save_path,
            epochs=epochs,
            device=device
        )

# end main

if __name__ == '__main__':
    main()