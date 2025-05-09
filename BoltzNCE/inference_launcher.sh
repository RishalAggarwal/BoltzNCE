#!/bin/bash
#SBATCH --job-name=ad2_inference
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --constraint=L40
#SBATCH --mail-user=jmc530@pitt.edu
#SBATCH --mail-type=ALL


replicates=1                                 # ← change this to 2, 3, 5, …
yaml_files=(
  saved_models/trained_vector_5_layer_ot_endpoint_tmax100_ema.yaml

)
######
num_configs=${#yaml_files[@]}
total_tasks=$(( num_configs * replicates ))

# must be 0 .. total_tasks-1
#SBATCH --array=0-$(( total_tasks - 1 ))

source activate boltznce
module load cuda/12.1

cfg_idx=$(( SLURM_ARRAY_TASK_ID / replicates ))
replicate=$(( SLURM_ARRAY_TASK_ID % replicates ))
config="${yaml_files[$cfg_idx]}"

echo "[${SLURM_ARRAY_TASK_ID}] cfg #$cfg_idx → $config   run #$replicate"
python infer_ad2.py \
  --config "$config" \
  --n_sample_batches 20 \
  --n_samples 500 \
  --wandb_inference_name "$(basename "$config" .yaml)_run${replicate}"
