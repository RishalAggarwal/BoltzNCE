#!/bin/bash

#SBATCH --job-name=aa2_inference
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --constraint=L40
#SBATCH --mail-user=jackychen@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-199%20           # 200 prefixes → indices 0 through 199

# list of 200 two‐letter prefixes
prefixes=(
  "EE" "TQ" "DV" "VM" "RR" "IP" "GY" "AG" "TL" "FW" "YY" "WP" "FD" "HM" "DE" "NS"
  "QN" "IC" "VC" "NN" "QS" "GW" "SR" "AI" "WM" "KP" "YK" "AF" "FV" "RS" "RN" "GE"
  "QA" "KM" "ED" "YV" "FK" "WL" "YE" "SS" "LN" "TC" "DY" "GV" "QR" "CI" "NR" "GK"
  "CT" "WQ" "SN" "LS" "FE" "HL" "NL" "VA" "YF" "SP" "AK" "LP" "EI" "WR" "GH" "IA"
  "QL" "DG" "IR" "CD" "MM" "LC" "TN" "WA" "YH" "AE" "HA" "EG" "SC" "GF" "MP" "CY"
  "RM" "DI" "VR" "GI" "DF" "NP" "HN" "FG" "LQ" "AW" "SL" "WS" "LL" "YG" "WN" "QP"
  "NM" "IN" "GG" "VS" "FI" "AY" "EF" "YT" "FT" "ML" "VN" "QV" "AL" "AQ" "EN" "CP"
  "QK" "IF" "DS" "VH" "RW" "FR" "LD" "KT" "YR" "QE" "IH" "GA" "DN" "CQ" "DA" "KF"
  "LK" "ER" "SV" "GS" "CL" "II" "QD" "VT" "FN" "HG" "EA" "KH" "SE" "WG" "YN" "IT"
  "QY" "YP" "KK" "SF" "TV" "FM" "VW" "GC" "MH" "CA" "YM" "WD" "LF" "FP" "HY" "WW"
  "SH" "AS" "FC" "VY" "GM" "QI" "VD" "QT" "EQ" "YC" "LH" "HW" "AA" "KW" "WE" "FQ"
  "LG" "VK" "DP" "DM" "RI" "MT" "SG" "EC" "YQ" "HE" "FL" "WK" "KY" "EP" "LI" "VE"
  "MG" "DC" "RG" "QH" "GL" "WV" "EM" "LT"
)

# pick out the prefix for this array task
prefix=${prefixes[$SLURM_ARRAY_TASK_ID]}
source activate boltznce
# run your inference with the selected prefix
python infer_aa2.py \
  --config configs/infer_vector_aa2_kabsch_ema.yaml \
  --wandb_inference_name inference_aa2_vector_${prefix}_100k \
  --peptide "${prefix}" \
  --save_generated \
  --save_prefix "./generated/train/" \
  --no-divergence \
  --data_path data/2AA-1-large/ \
  --data_directory train/ \
  --n_sample_batches 100 \
  --no-compute_metrics \


python infer_aa2.py \
  --config configs/infer_vector_aa2_kabsch_ema.yaml \
  --wandb_inference_name inference_aa2_vector_GG_100k \
  --peptide GG \
  --save_generated \
  --save_prefix "./generated/train/" \
  --no-divergence \
  --data_path data/2AA-1-large/ \
  --data_directory train/ \
  --n_sample_batches 100 \
  --no-compute_metrics \




