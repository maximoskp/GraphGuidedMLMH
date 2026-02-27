from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from models import ContrastiveSpaceModel
import pickle
import numpy as np
import os
from train_utils import train_contrastive
from data_utils import ContrastiveCollator
import evaluation_utils as eval_utils

batch_size = 4
device_name = 'cuda:0'
shared_dim = 32

if device_name == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)
# end device selection

val_path = 'data/contrastive_dataset/CA_test.pickle'

with open(val_path, 'rb') as f:
    val_dataset = pickle.load(f)

collator = ContrastiveCollator(pad_id=0)

for source_model in ['lstm', 'matrix', 'bot', 'graph']:
    source_key = f'{source_model}_embeddings'

    source_dim = val_dataset[0][source_key].shape[0]
    transformer_dim = val_dataset[0]['transformer_embeddings'].shape[0]

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = ContrastiveSpaceModel(source_dim, transformer_dim, shared_dim=shared_dim)
    model.to(device)

    checkpoint = torch.load(f'saved_models/contrastive/{source_model}.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f'\nResults for {source_model}:')
    eval_utils.evaluate_alignment(model, val_loader, source_key, device)

print('\n\n' + '='*30 + '\n\n')
print('OOD results ' + '-'*20)
print('\n\n' + '='*30 + '\n\n')

parent_dir_idioms = 'data/contrastive_dataset'
subdirs_dirs = os.listdir(parent_dir_idioms)
idiom_dirs = [f for f in subdirs_dirs]
# remove some idiom dirs that are not needed
idiom_dirs.remove('CA_test.pickle')
idiom_dirs.remove('CA_train.pickle')

full_dirs = []
for d in tqdm(idiom_dirs):
    full_dirs.append(os.path.join(parent_dir_idioms, d))

val_dataset = []
for i, idiom in enumerate(full_dirs):
    print(idiom)
    with open(idiom, 'rb') as f:
        dataset = pickle.load(f)
    for d in dataset:
        val_dataset.append(d)

for source_model in ['lstm', 'matrix', 'bot', 'graph']:
    source_key = f'{source_model}_embeddings'

    source_dim = val_dataset[0][source_key].shape[0]
    transformer_dim = val_dataset[0]['transformer_embeddings'].shape[0]

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = ContrastiveSpaceModel(source_dim, transformer_dim, shared_dim=shared_dim)
    model.to(device)

    checkpoint = torch.load(f'saved_models/contrastive/{source_model}.pt', map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f'\nResults for {source_model}:')
    eval_utils.evaluate_alignment(model, val_loader, source_key, device)
