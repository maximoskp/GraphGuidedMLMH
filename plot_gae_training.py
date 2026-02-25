import os
from tqdm import tqdm
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import plot_utils as pu
import embedding_utils as eu
import numpy as np
import pickle
from sklearn.decomposition import PCA

if not os.path.exists('data/gae_PCA_per_epoch.pickle'):
    val_dir = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'
    parent_dir_idioms = '/mnt/ssd2/maximos/data/coinvent_midi'
    other_dirs = os.listdir(parent_dir_idioms)
    idiom_dirs = [f for f in other_dirs if '.pickle' not in f]

    # full_dirs = [val_dir]
    full_dirs = []
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

    # first do a PCA on the last-epoch model to get the basis
    graph_embeds = []
    idiom_ids = []
    for i, idiom in enumerate(full_dirs):
        print('idiom: ', idiom)
        val_dataset = CSGridMLMDataset(idiom, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
        graph_np = eu.get_graph_saved_version_embeddings_for_dataset(
            val_dataset,
            tokenizer,
            199,
            hidden_dim=128,
            encoder_internal_dim=128
        )
        print('graph_np shape: ', graph_np.shape)
        graph_embeds.append(graph_np)
        idiom_ids.append( i*np.ones( (graph_np.shape[0], 1) ) )

    idiom_ids_np = np.vstack(idiom_ids)
    graph_embeds_np = np.vstack(graph_embeds)

    base_pca = PCA(n_components=2).fit(graph_embeds_np)

    graph_data = {
        'idiom_ids_np': idiom_ids_np,
        'base_pca': base_pca
    }

    for saved_version in tqdm(range(200)):
        graph_embeds = []
        for i, idiom in enumerate(full_dirs):
            # print(saved_version, '-' ,idiom)
            val_dataset = CSGridMLMDataset(idiom, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
            graph_np = eu.get_graph_saved_version_embeddings_for_dataset(
                val_dataset,
                tokenizer,
                saved_version,
                hidden_dim=128,
                encoder_internal_dim=128
            )
            graph_embeds.append(graph_np)
        graph_embeds_np = np.vstack(graph_embeds)
        graph_pca_np = base_pca.transform(graph_embeds_np)
        graph_data[f'epoch_{saved_version}'] = {
            'graph_pca_np': graph_pca_np
        }

    with open(f'data/gae_PCA_per_epoch.pickle', 'wb') as f:
        pickle.dump(graph_data, f)
# end if exists

print('loading graph PCA data...')
with open(f'data/gae_PCA_per_epoch.pickle', 'rb') as f:
    graph_data = pickle.load(f)

for saved_version in tqdm(range(200)):
    print(f'Plotting epoch {saved_version}...')
    data_tsne = graph_data[f'epoch_{saved_version}']['graph_pca_np']
    ids_np = graph_data['idiom_ids_np']
    pu.plot_idioms_2(data_tsne, ids_np, subfolder='gae_training', caption=f'epoch_{saved_version:0>3}', method='pca')