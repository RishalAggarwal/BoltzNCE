# BoltzNCE
Boltzmann emulator -> generator with EBM


## Setup Environment

All the required packages besides ```bgmol, bgflow``` are in ```install_things.sh```

Trained model weights have been uploaded, but to reproduce the next section is for model training.

## Dataset

The ADP dataset can be downloaded from [here](https://osf.io/srqg7/files/osfstorage?view_only=28deeba0845546fb96d1b2f355db0da5). Download the ```ad2_*.npy``` files and place them in the ```data``` folder.

The dipeptides dataset can be downloaded from [here](https://osf.io/n8vz3/files/osfstorage?view_only=1052300a21bd43c08f700016728aa96e#). Add the downloaded stuff to the ```data``` folder.

## Trained models

Trained models can be downloaded from [here](https://bits.csb.pitt.edu/files/BoltzNCE/saved_models_bits/). Add them to the ```saved_models``` directory

## ADP experiments

### Model training

```cd BoltzNCE```

For training the flow matching models on the original **unbiased** dataset run the following:

GVP - Vector field:

```python train_ad2.py --config ./configs/unweighted_ot_ema.yaml```

GVP - Endpoint:

```python train_ad2.py --config ./configs/unweighted_ot_endpoint_tmax100_ema.yaml```

For training the flow matching models on the **biased** dataset run the following:

GVP - Vector field:

```python train_ad2.py --config ./configs/train_vector_ot_ema.yaml```

GVP - Endpoint:

```python train_ad2.py --config ./configs/train_vector_endpoint_tmax100_ema.yaml```

The flow matching models need to be trained before we can set the energy based models to train. To train the Energy Based Models, run the following commands:

BoltzNCE - Vector field:

```python train_ad2.py --config ./configs/train_potential_graphormer_1b8ld256.yaml```

BoltzNCE - Endpoint:

```python train_ad2.py --config ./configs/train_potential_graphormer_endpoint.yaml```

### Model inference

To evaluate the flow matching models trained on the **unbiased** dataset:

GVP - Vector field:

```python infer_ad2.py --config saved_models/unweighted_ot_ema.yaml```

GVP - Endpoint

```python infer_ad2.py --config saved_models/unweighted_ot_endpoint_tmax100_ema.yaml```

To evaluate the flow matching models trained on **biased** dataset:

GVP - Vector field:

```python infer_ad2.py --config ./configs/infer_vector_ot_ema.yaml```

GVP - Endpoint:

```python infer_ad2.py --config ./configs/infer_vector_endpoint_tmax100_ema.yaml```

To evaluate the Energy based models run the following:

BoltzNCE - Vector field:

```python infer_ad2.py --config ./configs/infer_potential_graphormer_1b8ld256.yaml```

BoltzNCE - Endpoint:

```python infer_ad2.py --config ./configs/infer_potential_graphormer_endpoint.yaml```

## Dipeptide Experiments

### Model Training

To train the vector field model:

```python train_aa2.py --config configs/train_vector_kabsch_aa2.yaml```

To generate the dataset for EBM training with the trained vector field model, have a look at **sample_generation_aa2.sh**.

To train the EBM model:

```python train_aa2.py --config configs/train_potential_aa2_small_biased.yaml```

### Model Inference

To run inference on a sample dipeptide:

```python infer_aa2.py --config configs/infer_potential_aa2_small_correctedbias.yaml --no-divergence --wandb_inference_name inference_aa2_potential_small_correctedbias_{dipeptide}_100k --peptide {dipeptide} --n_sample_batches 200 --save_generated --save_prefix ./generated/{dipeptide}_ebm_100k_1_```

Fill in the two letter sequence for the dipeptide above

## Other evaluations/benchmarks

The remaining evaluations/benchmarks are present in the notebooks folder