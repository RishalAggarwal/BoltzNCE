#!/bin/bash
#SBATCH --job-name=ad2_inference
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --constraint=L40
#SBATCH --mail-user=jmc530@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-20%15

# ─── replicates ──────────────────────────────────────────────────────────────
replicates=5

# ─── parameter grid (largest → smallest, by decades) ─────────────────────────
rtol_vals=(1e-5)
atol_vals=(1e-5)
tmin_vals=(1e-3)

# ─── one or more model configs ───────────────────────────────────────────────
# yaml_files=(
#   saved_models/trained_vector_5_layer_ot.yaml
#   # add more if you like…
# )

yaml_files=( saved_models/trained_vector_5_layer_ot.yaml  

saved_models/trained_vector_5_layer_ot_ema.yaml  

saved_models/trained_vector_5_layer_ot_endpoint_tmax100.yaml  

saved_models/trained_vector_5_layer_ot_endpoint_tmax100_ema.yaml  

) 

# ─── derive sizes & total tasks ──────────────────────────────────────────────
num_cfg=${#yaml_files[@]}
len_r=${#rtol_vals[@]}
len_a=${#atol_vals[@]}
len_t=${#tmin_vals[@]}
grid_size=$(( len_r * len_a * len_t ))
total_tasks=$(( num_cfg * grid_size * replicates ))

# now set your SLURM array to cover [0 .. total_tasks-1]


# ─── which task are we? ──────────────────────────────────────────────────────
idx=$SLURM_ARRAY_TASK_ID

# split into grid‐index and replicate‐index
grid_idx=$(( idx / replicates ))
rep_idx=$(( idx % replicates ))

# pick which config file
cfg_idx=$(( grid_idx / grid_size ))
config=${yaml_files[$cfg_idx]}

# break the grid index into rtol/atol/tmin
rem=$(( grid_idx % grid_size ))
rtol_idx=$(( rem / (len_a * len_t) ))
rem2=$(( rem % (len_a * len_t) ))
atol_idx=$(( rem2 / len_t ))
tmin_idx=$(( rem2 % len_t ))

rtol=${rtol_vals[$rtol_idx]}
atol=${atol_vals[$atol_idx]}
tmin=${tmin_vals[$tmin_idx]}

echo "[#${idx}] cfg=$config, rtol=$rtol, atol=$atol, tmin=$tmin, rep=$rep_idx"

module load cuda/12.1
source activate boltznce

python infer_ad2.py \
  --config "$config" \
  --n_sample_batches 40 \
  --n_samples 500 \
  --wandb_inference_name "$(basename "$config" .yaml)_rtol${rtol}_atol${atol}_tmin${tmin}_rep${rep_idx}_div" \
  --rtol "$rtol" \
  --atol "$atol" \
  --tmin "$tmin" \
  
