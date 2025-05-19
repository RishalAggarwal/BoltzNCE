#!/bin/bash
# submit_all_ad2.sh
# -----------------
# Loops through ../configs/*.yaml and sbatch’s a job for each.

for cfg in ../configs/*.yaml; do
  # strip path and extension to get a short name
  name=$(basename "$cfg" .yaml)

  sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=ad2_training_${name}
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --constraint=L40
#SBATCH --mail-user=jmc530@pitt.edu
#SBATCH --mail-type=ALL

# (optional) directory setup, data staging, traps, etc.
# mkdir /scr/\${USER}_\${SLURM_JOB_ID}.dcb.private.net
# trap "rsync -avz * \${SLURM_SUBMIT_DIR}" EXIT

source activate boltznce
module load cuda/12.1

python train_ad2.py --config "$cfg"
EOF

  echo "Submitted ad2_training_${name}"
done
