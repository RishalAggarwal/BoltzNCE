#!/bin/bash

#job name
#SBATCH --job aa2_inference
#SBATCH --partition koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -w g021
#SBATCH --constraint L40
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
#python infer_aa2.py --config configs/infer_potential_aa2_small_biased.yaml --no-divergence --wandb_inference_name inference_aa2_potential_small_biased_NY_100k --peptide NY --n_sample_batches 200
#python infer_aa2.py --config configs/infer_potential_aa2_small.yaml --no-divergence --wandb_inference_name inference_aa2_potential_small_vector_large_GN_100k --peptide GN --n_sample_batches 200
python infer_aa2.py --config configs/infer_potential_aa2_small_biased.yaml --no-divergence --wandb_inference_name inference_aa2_potential_small_biased_NF_100k_bs512 --peptide NF --n_sample_batches 200
