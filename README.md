# MIX3R

## Create cluster environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)

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
    numba=0.60.0 \
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

Alternatively, create the environment with `environment.yml` file:

```bash
micromamba env create -f environment.yml
```

1. Activate and pack the environment:

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

## Containers

### Build status

[![Dockerfile lint](https://github.com/precimed/mix3r/actions/workflows/docker.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/docker.yml)
[![Container build](https://github.com/precimed/mix3r/actions/workflows/container_build.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/container_build.yml)
[![Container build push](https://github.com/precimed/mix3r/actions/workflows/container_build_push.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/container_build_push.yml)

### Dependencies on host system

In order to set up these resource, some software may be required

- [Singularity/SingularityCE](https://sylabs.io/singularity/) or [Apptainer](https://apptainer.org)
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.com)
- [ORAS CLI](https://oras.land)

### Docker

[Docker](https://www.docker.com/) is a platform for developing, shipping, and running applications in containers.
It allows you to package your application and all its dependencies into a single image that can be run on any machine.

#### Fetch from GitHub Container Registry

Docker containers with Mix3r are available on GitHub Container Registry. To pull the image, run the following command:

```bash
docker pull ghcr.io/precimed/mix3r:<tag>
```

where `<tag>` is the version of the image you want to pull from the [GitHub Container Registry](https://github.com/precimed/mix3r/pkgs/container/mix3r). The latest version is tagged as `latest`.

#### Build locally on host

To build the image locally, you need to have Docker installed on your machine. Then you can run the following command in the root directory of the repository:

```bash
    docker build --platform=linux/amd64 -t ghcr.io/precimed/mix3r -f docker/Dockerfile .
```

It will be tagged `latest` by default.

#### Running the container

You can also run the container with a command

```bash
    docker run ghcr.io/precimed/mix3r:latest <command>  # Available commands are: mix3r_int_weights, extract_p, make_template, make_euler, bash, python, ipython, jupyter
```

For interactive sessions (e.g., `bash`, `ipython`, `jupyter`), you can run the container with the following command:

```bash
    docker run -it --rm ghcr.io/precimed/mix3r:latest bash
```

### Singularity/Apptainer

[Singularity](https://sylabs.io/singularity/) and [Apptainer](https://apptainer.org) are container platforms that allow you to run containers on HPC systems.

#### Fetch from GitHub Container Registry

To obtain updated versions of the Singularity Image Format (.sif) container file `, issue

```bash
cd <path/to/>/mix3r/apptainer
mv mix3r.sif mix3r.sif.old  # optional, just rename the old(er) file if present
apptainer build -F --build-arg mix3r_image="docker://ghcr.io/precimed/mix3r:<tag>" mix3r.sif mix3r.def  # or
singularity build -F --build-arg mix3r_image="docker://ghcr.io/precimed/mix3r:<tag>" mix3r.sif mix3r.def # or 
oras pull ghcr.io/precimed/mix3r_sif:<tag> # recommended
```

where `<tag>` corresponds to a tag listed under [packages](https://github.com/precimed/mix3r/pkgs/container/mix3r_sif),
such as `latest`, `main`, or `sha_<GIT SHA>`. 
The `oras pull` statement pulls the `mix3r.sif` file from [ghcr.io](https://github.com/precimed/mix3r/pkgs/container/mix3r_sif) using the [ORAS](https://oras.land) registry, without the need to build the container locally.

#### Running the container

The mix3r container can be run with the following command(s):

```bash
export ROOT_DIR=<path/to/>/mix3r
export APPTAINER_BIND='/<path/to/data>:<path/to/data>:ro'
export SIF=$ROOT_DIR/apptainer/mix3r.sif
export CONFIG=$ROOT_DIR/examples/config.json
export MIX3R="apptainer run --nv ${SIF}"  # --nv is for CUDA/GPU support

$MIX3R mix3r_int_weights --config $CONFIG
$MIX3R extract_p --input my_analysis_run_*.json --out my_analysis_all_runs
$MIX3R make_euler my_analysis_all_runs.parameters.csv my_analysis.euler.png "Trait 1" "Trait 2" "Trait 3"
```

There are also other commands available in the container, such as `make_template`, `bash`, `python`, `ipython`, and `jupyter`.
