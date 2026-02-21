from data_utils import CSGridMLMDataset
from GridMLM_tokenizers import CSGridMLMTokenizer
import os

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

root_path = '/mnt/ssd2/maximos/data/coinvent_midi'

idioms = os.listdir(root_path)

for i in idioms:
    data_dir = os.path.join(root_path, i)
    print(i)
    train_dataset = CSGridMLMDataset(data_dir, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')