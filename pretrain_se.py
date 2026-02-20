import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader
from models import SEFiLMModel
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
from train_utils import train_with_curriculum

batchsize = 16
device_name = 'cuda:0'
lr = 1e-5
epochs = 200

train_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train'
val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'

def main():
    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    train_dataset = CSGridMLMDataset(train_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
    val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    loss_fn=CrossEntropyLoss(ignore_index=-100)

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
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/SE/', exist_ok=True)
    results_path = 'results/SE/' + 'pretraining.csv'

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/SE/', exist_ok=True)
    save_dir = 'saved_models/SE/'
    transformer_path = save_dir + 'pretrained.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        curriculum_type='f2f',
        epochs=epochs,
        exponent=5,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id
    )

# end main

if __name__ == '__main__':
    main()