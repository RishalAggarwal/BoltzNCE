#!/bin/bash

#job name
#SBATCH --job ad2_inference
#SBATCH --partition koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --exclude=g018
#SBATCH --constraint L40
##SBATCH -w g022
#SBATCH --mail-user=ria43@pitt.edu
#SBATCH --mail-type=ALL

# directory name where job will be run (on compute node)
#job_dir="${user}_${SLURM_JOB_ID}.dcb.private.net"

# creating directory on /scr folder of compute node
#mkdir /scr/$job_dir

# put date and time of starting job in a file
#date > date.txt

# put hostname of compute node in a file
#hostname > hostname.txt

# copy files on exit or interrupt
# make sure this is before your main program for it to always run on exit
#trap "echo 'copying files'; rsync -avz * ${SLURM_SUBMIT_DIR}" EXIT

# copy the submit file (and all other related files/directories)
#rsync -a ${SLURM_SUBMIT_DIR}/*.pkl /scr/${job_dir}


source activate BoltzNCE
module load cuda/12.1
#python ./train_pharmnn.py --train_data data/chemsplit_train0.pkl --test_data data/chemsplit_test0.pkl  --wandb_name default_chemsplit0_large_256 --grid_dimension 15.5  --expand_width 0 --model models/default_chemsplit0_large_256_last_model.pkl --lr 0.00001
#python ./train_pharmnn.py --train_data data/chemsplit_train2_with_ligand.pkl --test_data data/chemsplit_test2_with_ligand.pkl  --wandb_name obabel_chemsplit2_2 --negative_data data/obabel_chemsplit_2_negatives_train.txt --batch_size 256 --model models/obabel_chemsplit2_last_model.pkl --lr 0.00001
python infer_ad2.py --config ../configs/infer_vector_ot_ema.yaml --n_sample_batches 20 --n_samples 500 --wandb_inference_name vector_ot_ema_10k_inference
