### Create cluster environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)

The procedure should work on most of Linux distributions (but not on Mac).

1. Create a folder, download the latest version of micromamba, configure and [install](https://mamba.readthedocs.io/en/latest/installation.html):

```
mkdir -p micromamba/root
cd micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=${PWD}/root
eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba create -n mix3r \
    -c conda-forge \
    python=3.12 \
    conda-pack=0.8.0 \
    numpy=2.0.2 \
    scipy=1.14.1 \
    numba=0.6.0 \
    cuda-nvcc cuda-nvrtc "cuda-version>=12.0,<=12.4" \
    pandas=2.2.3 \
    matplotlib-base=3.9.2 \
    ipython=8.28.0 \
    notebook=7.2.2 \
    r=4.4 \
    r-essentials=4.4 \
    r-irkernel=1.3.2 \
    r-eulerr=7.0.2 \
    -c numba \
    icc_rt=2020.2 \
    -c rapidsai \
    pynvjitlink=0.3.0 \
    --yes
```
This assumes that you have NVIDIA driver supporting CUDA toolkit version 12. If your driver is for CUDA 11, you have to install CUDA `cudatoolkit` instead of `cuda-nvcc` and `cuda-nvrtc`, see [numba's documentation](https://numba.readthedocs.io/en/stable/cuda/overview.html#software) for more details. For compatibility of minor versions (i.e. 12.4 vs 12.6) it might be better to set the upper bound of cuda-version (which is equal to 12.4 in the code above, and defines the version of CUDA toolkit which will be installed) to the version supported by the NVIDIA driver. CUDA toolkit version supported by the installed NVIDIA driver can be checked with `nvidia-smi` command (shown in the top row of the output table, after the driver version).

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

### To run on TSD

1. Prepare config files (you may use `config_scz_adhd_mig_aug22_1.json` as template) for multiple (e.g. 16) runs with different variant subsets. You may do it by changing `"rand_prune_seed"` parameter in the config file, e.g. set `"rand_prune_seed" : i` for the i-th run (also don't forget to use different output files for all runs, e.g. for the i-th run set `"out" : "my_analysis_run_i.json"`).
2. Submit jobs for all configs to cluster:

```
for i in {1..16}; do sbatch run_slurm_int.sh path/to/your_config_${i}.json; done
```

3. When all runs are finished, produce a table with estimated model parameters for all runs:

```
python extract_p.py --input my_analysis_run_*.json --out my_analysis_all_runs
```

`my_analysis_all_runs.parameters.csv` file will be created.
You can also produce a table for a selection of runs:

```
python extract_p.py --input my_analysis_run_1.json my_analysis_run_3.json my_analysis_run_7.json --out my_analysis_selected_runs
```

4. Plot "median" Euler diagram:

```
Rscript make_euler.r my_analysis_all_runs.parameters.csv my_analysis.euler.png "Trait 1" "Trait 2" "Trait 3"  
```

Where the first argument is a path to the file with model parameters produced with `extract_p.py` script, the second argument is output file name and the last three arguments are trait labels to be used in the Euler diagram.
