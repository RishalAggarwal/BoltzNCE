set -e
conda install numpy=1.26.4 -y
pip install einops
conda install pytorch=2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
conda install matplotlib -y
conda install -c dglteam/label/th22_cu121 dgl -y
conda install -c conda-forge wandb -y
conda install -c conda-forge fsspec -y
pip install torchdata==0.7.1c
conda install -c conda-forge mdtraj=1.9.9 -y
conda install -c conda-forge openmm=8.1.1 cuda-version=12.1 -y
conda install openmmtools -y
conda install xtb-python -y
pip install torchdiffeq
pip install POT
pip install ema-pytorch