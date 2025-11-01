Installation instructions
## Step 1: Create a python environment conda (+ install basic libraries)
```bash
conda create -n "hbird" python=3.12 ipython pylint ipykernel 
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

## Step 4: Install the libraries (Multiple Examples) 
### Use faiss-gpu for Nerarest Neighbor Retrieval
#### Approach 1: Use CUDA 11.8
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0

pip install lightning==2.5.5
pip install tqdm==4.67.1
pip install numpy==1.26.4
pip install scipy==1.11.4
```
You could also find the singularity definition of this approach at [hbird_cuda11_8.def](./singularity_defs/hbird_cuda11_8.def)

#### Approach 2: Use CUDA 12.1
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install faiss-gpu-cu12

pip install lightning==2.5.5
pip install tqdm==4.67.1
pip install numpy==1.26.4
pip install scipy==1.11.4
```
You could also find the singularity definition of this approach at [hbird_cuda12_1.def](./singularity_defs/hbird_cuda12_1.def)

#### Approach 3: Use CUDA 12.6
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu126

conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs "cuda-version>=12.6,<12.7" "numpy>=2.0,<2.3" "scipy>=1.14,<1.16"

pip install lightning==2.5.5
pip install tqdm==4.67.1
```
You could also find the singularity definition of this approach at [hbird_cuda12_6.def](./singularity_defs/hbird_cuda12_6.def)

### Use scann (on cpu)for Nerarest Neighbor Retrieval
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu126

pip install faiss-cpu
pip install scann

pip install lightning==2.5.5
pip install tqdm==4.67.1
```
You could also find the singularity definition of this approach at [hbird_cpu.def](./singularity_defs/hbird_cpu.def)


