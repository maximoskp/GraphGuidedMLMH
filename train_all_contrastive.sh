#!/bin/bash

# List of Python scripts with their respective arguments
scripts=(
    "python train_contrastive.py -s lstm -d 64 -g0 -e 100 -l 1e-5 -b 32"
    "python train_contrastive.py -s matrix -d 64 -g0 -e 100 -l 1e-5 -b 32"
    "python train_contrastive.py -s bot -d 64 -g0 -e 100 -l 1e-5 -b 32"
    "python train_contrastive.py -s graph -d 64 -g0 -e 100 -l 1e-5 -b 32"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source /opt/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done