# BoltzNCE
Boltzmann emulator -> generator with NCE


## Setup Environment

All the required packages besides ```bgmol, bgflow``` are in ```install_things.sh```

Trained model weights have been uploaded, but to reproduce the next section is for model training.

## Dataset

The dataset can be downloaded from [here](https://osf.io/srqg7/files/osfstorage?view_only=28deeba0845546fb96d1b2f355db0da5). Download the ```ad2_*.npy``` files and place them in the ```data``` folder.

## Model training

```cd BoltzNCE```

For training the flow matching models on the original unbiased dataset run the following:

GVP - Vector field:

```python train_ad2.py --config ../configs/unweighted_ot_ema.yaml```

GVP - Endpoint:

```python train_ad2.py --config ../configs/unweighted_ot_endpoint_tmax100_ema.yaml```

For training the flow matching models on the biased dataset run the following:

GVP - Vector field:

```python train_ad2.py --config ../configs/train_vector_ot_ema.yaml```

GVP - Endpoint:

```python train_ad2.py --config ../configs/train_vector_ot_endpoint_tmax100_ema.yaml```

The flow matching models need to be trained before we can set the energy based models to train. To train the Energy Based Models, run the following commands:

BoltzNCE - Vector field:

```python train_ad2.py --config ../configs/train_potential_graphormer_1b8ld256.yaml```

BoltzNCE - Endpoint:

```python train_ad2.py --config ../configs/train_potential_graphormer_endpoint.yaml```

## Model inference

To evaluate the flow matching models trained on the unbiased dataset:

GVP - Vector field:

```python infer_ad2.py --config saved_models/unweighted_ot_ema.yaml```

GVP - Endpoint

```python infer_ad2.py --config saved_models/unweighted_ot_endpoint_tmax100_ema.yaml```

To evaluate the flow matching models trained on biased dataset:

GVP - Vector field:

```python infer_ad2.py --config ../configs/infer_vector_ot_ema.yaml```

GVP - Endpoint:

```python infer_ad2.py --config ../configs/infer_vector_endpoint_tmax100_ema.yaml```

To evaluate the Energy based models run the following:

BoltzNCE - Vector field:

```python infer_ad2.py --config ../configs/infer_potential_graphormer_1b8ld256.yaml```

BoltzNCE - Endpoint:

```python infer_ad2.py --config ../configs/infer_potential_graphormer_endpoint.yaml```

 