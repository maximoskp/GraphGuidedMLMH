import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader
from models import SEFiLMModel, ContrastiveSpaceModel, contrastive_normalized_loss
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_lacta
from data_utils import ContrastiveCollator
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
    print('shared', ' - shared_dim: ', shared_dim)

    collator = ContrastiveCollator(pad_id=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    contrastive_model = ContrastiveSpaceModel(source_dim, transformer_dim, shared_dim=shared_dim)
    checkpoint = torch.load(f'saved_models/contrastive/{source_name}.pt', map_location=device_name)
    contrastive_model.load_state_dict(checkpoint)
    contrastive_model.to(device)

    contrastive_loss_fn = contrastive_normalized_loss

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    logits_loss_fn =CrossEntropyLoss(ignore_index=-100)

    transformer_model = SEFiLMModel(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=512,
        nhead=8,
        num_layers=8,
        grid_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
        guidance_dim=shared_dim,
        device=device,
    )
    checkpoint = torch.load('saved_models/SE/pretrained.pt', map_location=device_name)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    optimizer = AdamW(transformer_model.film_parameters(), lr=lr)

    # save results
    results_path = os.path.join( 'results', 'lacta', f'{source_name}.csv' )
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/lacta', exist_ok=True)

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/lacta/', exist_ok=True)
    save_dir = 'saved_models/lacta/'
    transformer_path = save_dir + f'{source_name}.pt'

    train_lacta(
        transformer_model, contrastive_model, 
        contrastive_loss_fn, logits_loss_fn,
        optimizer, train_loader, val_loader, tokenizer.mask_token_id,
        source_key,
        epochs=epochs,
        exponent=-1,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=1,
        tqdm_position=0
    )

# end main

if __name__ == '__main__':
    main()