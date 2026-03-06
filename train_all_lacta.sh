#!/bin/bash

# TODO:
# 1) Correct the transformer model so that it isolates the FiLM
# parameters in a ModuleList.
# 2) Add an argument for selecting contrastive vs MSE loss.

# List of Python scripts with their respective arguments
scripts=(
    "train_lacta.py -s lstm -d 64 -g0 -e 200 -l 1e-4 -b 32"
    "train_lacta.py -s matrix -d 64 -g0 -e 200 -l 1e-4 -b 32"
    "train_lacta.py -s bot -d 64 -g0 -e 200 -l 1e-4 -b 32"
    "train_lacta.py -s graph -d 64 -g0 -e 200 -l 1e-4 -b 32"
)

# Name of the conda environment
conda_env="torch"

# Path to global conda
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"

for script in "${scripts[@]}"; do
    script_name=$(echo "$script" | awk '{print $1}')
    screen_name=$(basename "$script_name" .py)

    screen -dmS "$screen_name" bash -c "
        source \"$CONDA_SH\"
        conda activate \"$conda_env\"
        python $script
        exec bash
    "

    echo "Started screen '$screen_name' for script '$script'."
done