# MIX3R

This repository contains the source code for the trivariate mixer method to disentangle the genetic overlap between three phenotypes using summary statistics from genome-wide association studies (GWAS).

If you find this code useful, please cite the paper:

    Dissecting the genetic overlap between three complex phenotypes with trivariate MiXeR
    Alexey A. Shadrin, Guy Hindley, Espen Hagen, Nadine Parker, Markos Tesfaye, Piotr Jaholkowski, Zillur Rahman, Gleda Kutrolli, Vera Fominykh, Srdjan Djurovic, Olav B. Smeland, Kevin S. O’Connell, Dennis van der Meer, Oleksandr Frei, Ole A. Andreassen, Anders M. Dale
    medRxiv 2024.02.23.24303236; doi: https://doi.org/10.1101/2024.02.23.24303236

Bibtex:

```bibtex
@article {Shadrin2024.02.23.24303236,
	author = {Shadrin, Alexey A. and Hindley, Guy and Hagen, Espen and Parker, Nadine and Tesfaye, Markos and Jaholkowski, Piotr and Rahman, Zillur and Kutrolli, Gleda and Fominykh, Vera and Djurovic, Srdjan and Smeland, Olav B. and O{\textquoteright}Connell, Kevin S. and van der Meer, Dennis and Frei, Oleksandr and Andreassen, Ole A. and Dale, Anders M.},
	title = {Dissecting the genetic overlap between three complex phenotypes with trivariate MiXeR},
	elocation-id = {2024.02.23.24303236},
	year = {2024},
	doi = {10.1101/2024.02.23.24303236},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Comorbidities are an increasing global health challenge. Accumulating evidence suggests overlapping genetic architectures underlying comorbid complex human traits and disorders. The bivariate causal mixture model (MiXeR) can quantify the polygenic overlap between complex phenotypes beyond global genetic correlation. Still, the pattern of genetic overlap between three distinct phenotypes, which is important to better characterize multimorbidities, has previously not been possible to quantify. Here, we present and validate the trivariate MiXeR tool, which disentangles the pattern of genetic overlap between three phenotypes using summary statistics from genome-wide association studies (GWAS). Our simulations show that the trivariate MiXeR can reliably reconstruct different patterns of genetic overlap. We further demonstrate how the tool can be used to estimate the proportions of genetic overlap between three phenotypes using real GWAS data, providing examples of complex patterns of genetic overlap between diverse human traits and diseases that could not be deduced from bivariate analyses. This contributes to a better understanding of the etiology of complex phenotypes and the nature of their relationship, which may aid in dissecting comorbidity patterns and their biological underpinnings.Availability and implementation The trivariate MiXeR tool and auxiliary scripts, including source code, documentation and examples of use are available at https://github.com/precimed/mix3rCompeting Interest StatementSrdjan Djurovic has received speaker{\textquoteright}s Honoria from Lundbeck. Anders M. Dale is a Founder of and holds equity in CorTechs Labs, Inc, and serves on its Scientific Advisory Board. He is also a member of the Scientific Advisory Board of Human Longevity, Inc. (HLI), and the Mohn Medical Imaging and Visualization Centre in Bergen, Norway. He receives funding through a research agreement with General Electric Healthcare (GEHC). The terms of these arrangements have been reviewed and approved by the University of California, San Diego in accordance with its conflict-of-interest policies. Ole A. Andreassen has received speaker fees from Lundbeck, Janssen, Otsuka, and Sunovion and is a consultant to Cortechs.ai and Precision Health AS.Funding StatementThis work was supported by the Research Council of Norway (grants: 223273, 326813, 273291, 273446, 334920, 324499, 324252, 223273, 300309 and 248778). The European Economic Area and Norway Grants (grants: EEA-RO-NO-2018-0535 and EEA-RO-NO-2018-0573). European Union{\textquoteright}s Horizon 2020 Research and Innovation Programme (grants: 847776, 801133 and 964874). National Institutes of Health: U24DA041123, U24DA055330, 5R01MH124839-02 and R01MH123724-01. South-Eastern Norway Regional Health Authority (grant 2022073). KG Jebsen Stiftelsen, The South-East Norway Regional Health Authority (grant 2022-087).Author DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:The individual-level genotyping data for the current study were obtained from UK Biobank under accession number 27412. All researchers who wish to access this resource must register with UK Biobank by completing the registration form in the Access Management System. GWAS summary statistics data used in this study are publicly available and corresponding links are provided in the manuscript.I confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) outside the research group so cannot be used to identify individuals.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).YesI have followed all appropriate research reporting guidelines, such as any relevant EQUATOR Network research reporting checklist(s) and other pertinent material, if applicable.YesAll data produced in the present work are contained in the manuscript. https://github.com/precimed/mix3r},
	URL = {https://www.medrxiv.org/content/early/2024/02/27/2024.02.23.24303236},
	eprint = {https://www.medrxiv.org/content/early/2024/02/27/2024.02.23.24303236.full.pdf},
	journal = {medRxiv}
}
```

## Installation and set up

To obtain the source code, clone the repository using [Git](https://git-scm.com/):

```bash
<path/to/repositories>
git clone https://www.github.com/precimed/mix3r
```

### Create cluster environment with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)

The procedure should work on most of Linux distributions (but not on Macs with ARM-based chipsets - M1, M2, M3, etc.).

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
    numba=0.61.2 \
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

#### To run on TSD

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

### Containers

#### Build status

[![Dockerfile lint](https://github.com/precimed/mix3r/actions/workflows/docker.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/docker.yml)
[![Container build](https://github.com/precimed/mix3r/actions/workflows/container_build.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/container_build.yml)
[![Container build push](https://github.com/precimed/mix3r/actions/workflows/container_build_push.yml/badge.svg)](https://github.com/precimed/mix3r/actions/workflows/container_build_push.yml)

#### Dependencies on host system

In order to set up these resource, some software may be required

- [Singularity/SingularityCE](https://sylabs.io/singularity/) or [Apptainer](https://apptainer.org)
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.com)
- [ORAS CLI](https://oras.land)

#### Docker

[Docker](https://www.docker.com/) is a platform for developing, shipping, and running applications in containers.
It allows you to package your application and all its dependencies into a single image that can be run on any machine.

##### Fetch from GitHub Container Registry

Docker containers with Mix3r are available on GitHub Container Registry. To pull the image, run the following command:

```bash
docker pull ghcr.io/precimed/mix3r:<tag>
```

where `<tag>` is the version of the image you want to pull from the [GitHub Container Registry](https://github.com/precimed/mix3r/pkgs/container/mix3r). The latest version is tagged as `latest`.

##### Build locally on host

To build the image locally, you need to have Docker installed on your machine. Then you can run the following command in the root directory of the repository:

```bash
    docker build --platform=linux/amd64 -t ghcr.io/precimed/mix3r -f docker/Dockerfile .
```

It will be tagged `latest` by default.

##### Running the container

You can also run the container with a command

```bash
    docker run ghcr.io/precimed/mix3r:latest <command>  # Available commands are: mix3r_int_weights, extract_p, make_template, make_euler, bash, python, ipython, jupyter
```

For interactive sessions (e.g., `bash`, `ipython`, `jupyter`), you can run the container with the following command:

```bash
    docker run -it --rm ghcr.io/precimed/mix3r:latest bash
```

#### Singularity/Apptainer

[Singularity](https://sylabs.io/singularity/) and [Apptainer](https://apptainer.org) are container platforms that allow you to run containers on HPC systems.

##### Fetch from GitHub Container Registry

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

##### Running the container

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
