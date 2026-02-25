import os
from tqdm import tqdm
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import plot_utils as pu
import embedding_utils as eu
import numpy as np
import pickle

val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'
parent_dir_idioms = '/mnt/ssd2/maximos/data/coinvent_midi'
other_dirs = os.listdir(parent_dir_idioms)
idiom_dirs = [f for f in other_dirs if '.pickle' not in f]

full_dirs = [val_dir]
for d in tqdm(idiom_dirs):
    full_dirs.append(os.path.join(parent_dir_idioms, d))

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

val_dataset = CSGridMLMDataset(val_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')

graph_data = {}
idiom_ids = []
graph_embeds = []

for saved_version in range(200):
    for i, idiom in enumerate(full_dirs):
        print(saved_version, '-' ,idiom)
        graph_np = eu.get_graph_saved_version_embeddings_for_dataset(
            val_dataset,
            tokenizer,
            saved_version,
            hidden_dim=128,
            encoder_internal_dim=128
        )
        graph_embeds.append(graph_np)
        idiom_ids.append( i*np.ones( (graph_np.shape[0], 1) ) )
    idiom_ids_np = np.vstack(idiom_ids)
    graph_embeds_np = np.vstack(graph_embeds)
    graph_data[saved_version] = {
        'graph_embeds_np': graph_embeds_np,
        'idiom_ids_np': idiom_ids_np
    }

with open(f'data/gae_training_epoch_data.pickle', 'wb') as f:
    pickle.dump(graph_data, f)