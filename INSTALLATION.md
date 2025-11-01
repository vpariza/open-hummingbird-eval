Installation instructions
```bash
module load python/3.12-conda
module load cuda/12.1.1
```
## Step 1: Create a python environment conda or just a python virtual environment
```bash
conda create -n "hbird" python=3.11 ipython
```

## Step 2: Activate the conda environment
```bash
conda activate hbird
```
or 
```bash
source activate hbird
```

## Step 3: Update pip
```bash
python -m pip install --upgrade pip
```

Other
```bash
# for CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# or install if faiss with NVIDIA cuVS is needed: then we need CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
```bash
pip install lightning==2.4.0
pip install torchmetrics==1.7.0
pip install tqdm==4.67.1 # optional
pip install scipy==1.15.2
pip install joblib==1.4.2
pip install numpy==1.26.4
pip install triton==2.2.0

# if using CUDA 12.1
pip install faiss-gpu-cu12
# if using NVIDIA cuVS
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0
# or/and
pip install scann
```

