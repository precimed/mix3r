### Create cluster environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
The procedure should work on most of Linux distributions (but not on Mac).
1. Create a folder, download the latest version of micromamba, configure and [install](https://mamba.readthedocs.io/en/latest/installation.html):
```
mkdir -p micromamba/root
cd micromamba
wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=${PWD}/root
eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba create -n mix3r -c conda-forge python=3.11 conda-pack numpy scipy numba pandas matplotlib-base ipython notebook -c numba icc_rt -c r r r-essentials r-irkernel r-eulerr
```
2. Activate and pack the environment:
```
micromamba activate mix3r
conda-pack -p root/envs/mix3r -o mix3r.tar.gz
```
3. Then copy `mix3r.tar.gz` to cluster and unpack:
```
mkdir mix3r
tar -zxf mix3r.tar.gz -C mix3r/
source mix3r/bin/activate
conda-unpack
```
