### Create cluster environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)

The procedure should work on most of Linux distributions (but not on Mac).

1. Create a folder, download the latest version of micromamba, configure and [install](https://mamba.readthedocs.io/en/latest/installation.html):

```
mkdir -p micromamba/root
cd micromamba
wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=${PWD}/root
eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate
micromamba create -n mix3r \
    -c conda-forge \
    python=3.11 \
    conda-pack=0.7.1 \
    numpy=1.26.0 \
    scipy=1.11.3 \
    numba=0.58.1 \
    pandas=2.1.1 \
    matplotlib-base=3.8.0 \
    ipython=8.16.1 \
    notebook=7.0.6 \
    r=4.3 \
    r-essentials=4.3 \
    r-irkernel=1.3.2 \
    r-eulerr=7.0.0 \
    -c numba \
    icc_rt=2020.2 \
    --yes
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
