set -e
mamba install numpy=1.26.4 -y
pip install einops
mamba install pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
mamba install matplotlib -y
mamba install -c dglteam/label/th22_cu121 dgl -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge fsspec -y
pip install torchdata==0.7.1c
mamba install -c conda-forge mdtraj=1.9.9 -y
mamba install -c conda-forge openmm=8.1.1 cuda-version=12.1 -y
mamba install openmmtools -y
mamba install xtb-python -y
pip install torchdiffeq
pip install POT
pip install ema-pytorch
mamba install -c conda-forge deeptime -y
pip install torchdyn
mamba install -c conda-forge mdanalysis -y