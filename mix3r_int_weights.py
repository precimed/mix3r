import numpy as np
import numba as nb
import pandas as pd
from numba import cuda
import math
from scipy.optimize import minimize, direct, brute, minimize_scalar, differential_evolution, NonlinearConstraint
from datetime import datetime
import os
import sys
import json
import argparse

COL_DTYPE = dict(SNP='U',
                 N='f4',
                 Z='f4', 
                 INFO='f4',
                 A1='U',
                 A2='U') # only SNP and trait-specific columns, other columns (CHR, BP, MAF) are taken from template


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='default_config.json', help="Path to configuration file.")
    return parser.parse_args(args)


def load_config(config_fname):
    with open(config_fname) as f:
        config = json.load(f)
    return config

@nb.njit
def get_r2_het(r2, r2_idx, het):
    r2_het = np.empty_like(r2)
    for i in range(len(r2)):
        r2_het[i] = r2[i]*het[r2_idx[i]]
    return r2_het

@nb.njit
def get_ld_n_idx(ld_n):
    # ld_n: ld_n[i] = size of i-th LD block
    # Returns:
    #     ld_n_idx: ld_n_idx[i] = index of the first element of the i-th LD block in corersponding r2 and r2_idx arrays
    ld_n_idx = np.zeros(len(ld_n)+1, dtype='i8') # indices of LD blocks in r2/r2_idx vectors
    for i, n_in_ld in enumerate(ld_n):
        ld_n_idx[i+1] = ld_n_idx[i] + n_in_ld # cumsum
    return ld_n_idx

@nb.njit
def get_r2_het_hist(b2use, r2_het, ld_n, nbin):
    n2use = b2use.sum() # number of used SNPs
    r2_het_hist = np.zeros(n2use*nbin, dtype='f4')
    # r2_het_hist_edges: max = 0.5, min = 0.5/NBIN_R2_HET_HIST
    r2_het_hist_edges = np.linspace(0.5/nbin, 0.5, nbin).astype(np.float32)
    i_r2orig = 0
    i_template2use = 0
    for i_template, nld in enumerate(ld_n):
        if b2use[i_template]:
            for x_r2_het in r2_het[i_r2orig:i_r2orig+nld]:
                for i, hist_edge in enumerate(r2_het_hist_edges):
                    if x_r2_het <= hist_edge:
                        r2_het_hist[i_template2use*nbin + i] += 1
                        break
            i_template2use += 1
        i_r2orig += nld
    return r2_het_hist

@nb.njit
def get_ld_scores(b2use, r2_het, ld_n, nbin):
    n2use = b2use.sum() # number of used SNPs
    ld_scores = np.zeros(n2use, dtype='f4')
    i_r2orig, i_template2use = 0, 0
    for i_template, nld in enumerate(ld_n):
        if b2use[i_template]:
            ld_scores[i_template2use] = sum(r2_het[i_r2orig:i_r2orig+nld])
            i_template2use += 1
        i_r2orig += nld
    return ld_scores

@nb.njit
def prune(pval, r2, r2_idx, ld_n, b2use, r2_thresh):
    # b2use: bool vector of SNP indices in template to consider for pruning. All indices not in i2use will not survive.
    # Returns:
    #     is_survived: bool vector, True if SNP survives pruning
    assert len(pval) == len(ld_n)
    assert len(b2use) == len(ld_n)
    isort = np.argsort(pval)
    is_survived = b2use[:]
    ld_n_idx = get_ld_n_idx(ld_n)
    for i in isort:
        if is_survived[i]:
            for j in range(ld_n_idx[i], ld_n_idx[i+1]): # j = index in r2/r2_idx vector
                if r2[j] > r2_thresh:
                    is_survived[r2_idx[j]] = False
            is_survived[i] = True
    return is_survived


@nb.njit
def get_total_het_used_chr(b2use, r2_idx, het, ld_n):
    n_use = np.zeros(len(b2use), dtype='i4') # n_use[i] = number of times i-th SNP on chromosome from template was in LD with any SNP used for fitting
    i_r2orig = 0
    for i_template, nld in enumerate(ld_n):
        if b2use[i_template]:
            n_use[r2_idx[i_r2orig:i_r2orig+nld]] += 1
        i_r2orig += nld
    # (n_use != 0).sum() = number of SNPs from template appeared at least once in LD with any SNP used for fitting
    # (n_use*het).sum() = sum of het of SNPs in all LD of all SNPs used for fiting
    # n_use.sum() == number of SNPs in all LD blocks of all SNPs used for fitting
    total_n_used = (n_use!=0).sum()
    total_het_used = het[n_use!=0].sum()
    return total_het_used, total_n_used

def get_total_het_used(template_dir, snps_df, rand_prune_seed, r2_prune_thresh):
    rng = np.random.default_rng(rand_prune_seed)
    total_het, total_n = 0, 0
    for chrom in snps_df.CHR.unique():
        ld_r2_file = os.path.join(template_dir, f'chr{chrom}.ld_r2')
        ld_idx_file = os.path.join(template_dir, f'chr{chrom}.ld_idx')
        r2 = np.memmap(ld_r2_file, dtype='f4', mode='r')
        r2_idx = np.memmap(ld_idx_file, dtype='i4', mode='r')
        snps_df_chr = snps_df.loc[snps_df.CHR == chrom,:]
        het = 2*snps_df_chr.MAF.values*(1 - snps_df_chr.MAF.values)
        ld_n = snps_df_chr.LD_N.values
            
        b2use = snps_df_chr.IS_VALID.values
        rand_pval = rng.random(snps_df_chr.shape[0])
        bpruned = prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
        total_het_chr, total_n_chr = get_total_het_used_chr(bpruned, r2_idx, het, ld_n)
        total_het += total_het_chr
        total_n += total_n_chr
    total_het_used = len(snps_df)*total_het/total_n
    return total_het_used


def swap_z_sign(snps_df, n):
    # Change snps_df inplace. Check allele correspondence between template and sumstats.
    # Set IS_VALID = False for SNPs with allele-mismatch.
    # Swap Z sign for SNPs with swapped alleles.
    # Drop A1_i and A2_i columns (alleles loaded from sumstats).
    
    # Make reverse compliments of reference alleles
    compliments = str.maketrans("ATGC","TACG")
    make_rev_comp = lambda x: x.translate(compliments)[::-1]
    a1_rev_comp = snps_df.A1.apply(make_rev_comp)
    a2_rev_comp = snps_df.A2.apply(make_rev_comp)
    
    for i in range(n):
        z_col, a1_col, a2_col = f"Z_{i}", f"A1_{i}", f"A2_{i}"
        # check if A1 in template is A1 or A2 (both possibly reverse compliment) in sumstats,
        # if nither fits the SNP is invalid.
        i_a1_is_a1 = (snps_df.A1 == snps_df[a1_col]) & (snps_df.A2 == snps_df[a2_col])
        i_a1_is_a1 |= (a1_rev_comp == snps_df[a1_col]) & (a2_rev_comp == snps_df[a2_col])
        i_a1_is_a2 = (snps_df.A1 == snps_df[a2_col]) & (snps_df.A2 == snps_df[a1_col])
        i_a1_is_a2 |= (a1_rev_comp == snps_df[a2_col]) & (a2_rev_comp == snps_df[a1_col])
        # SNP is valid if its alleles match either directly (A1-A1) or swapped (A1-A2), both possibly with reverse compliment
        snps_df.IS_VALID &= i_a1_is_a1 | i_a1_is_a2
        snps_df.loc[i_a1_is_a2, z_col] *= -1 # if A1 in template is A2 in sumstats swap sign of Z
        snps_df.drop(columns=[a1_col, a2_col], inplace=True)
        
        
def load_snps(template_dir, sumstats, *, chromosomes=range(1,23),
              z_thresh=None, info_thresh=None, maf_thresh=None, exclude_regions=[]):
    # Load template SNPs for given chromosomes.
    # Load (multiple) sumstats.
    # Merge each sumstats with template.
    # Allign alleles in sumstats to template and swap effect direction correspondingly.
    # Add IS_VALID column which is True for SNPs for all SNPs passing specified filtering
    if isinstance(sumstats, str):
        sumstats = [sumstats]
        
    # Read template SNPs
    print(f"Reading template SNPs for {len(chromosomes)} chromosomes from {template_dir}")
    snps_df_list = []
    for chrom in chromosomes:
        snp_file = os.path.join(template_dir, f'chr{chrom}.snp.gz')
        df = pd.read_table(snp_file, dtype={"CHR":'i4',"MAF":'f4',"LD_N":'i4'})
        snps_df_list.append(df)
    snps_df = pd.concat(snps_df_list, ignore_index=True)
    print(f"    {snps_df.shape[0]} SNPs")
    snps_df["IS_VALID"] = True
    # Read sumstats
    for i, fname in enumerate(sumstats):
        print(f"Loading sumstats from {fname}")
        cols = pd.read_table(fname, nrows=0).columns
        usecols = [c for c in cols if c in COL_DTYPE]
        df = pd.read_table(fname, usecols=usecols, dtype=COL_DTYPE)
        df.drop_duplicates(subset=["SNP"], keep='first', inplace=True)
        print(f"    {df.shape[0]} SNPs")
        col_rename = {c:f"{c}_{i}" for c in usecols if c!="SNP"}
        df.rename(columns=col_rename, inplace=True)
        snps_df = pd.merge(snps_df, df, on="SNP", how="left")
        snps_df.IS_VALID &= snps_df[f"Z_{i}"].notna() & snps_df[f"N_{i}"].notna()
    print(f"{snps_df.IS_VALID.sum()} common SNPs")
    
    # swap Z_i signs of z-scores and IS_VALID = False when alleles do not correspond to reference
    swap_z_sign(snps_df, len(sumstats))
    print(f"{snps_df.IS_VALID.sum()} SNPs with matched alleles")
    
    # Apply filters
    if z_thresh:
        z_cols = [c for c in snps_df.columns if c.startswith("Z_")]
        snps_df.IS_VALID &= (snps_df[z_cols].abs() < z_thresh).all(axis="columns")
        print(f"{snps_df.IS_VALID.sum()} SNPs with Z < {z_thresh}")
    if info_thresh:
        info_cols = [c for c in snps_df.columns if c.startswith("INFO_")]
        snps_df.IS_VALID &= (snps_df[info_cols] > info_thresh).all(axis="columns")
        print(f"{snps_df.IS_VALID.sum()} SNPs with INFO > {info_thresh}")
    if maf_thresh:
        snps_df.IS_VALID &= snps_df.MAF > maf_thresh
        print(f"{snps_df.IS_VALID.sum()} SNPs with MAF > {maf_thresh}")
    for region in exclude_regions:
        chrom, start_end = region.split(":")
        chrom = int(chrom)
        start, end = map(int, start_end.split("-"))
        i_drop = (snps_df.CHR == chrom) & (snps_df.BP > start) & (snps_df.BP < end)
        snps_df.IS_VALID &= ~i_drop
        print(f"    {i_drop.sum()} SNPs excluded from {region}")
    print(f"{snps_df.IS_VALID.sum()} SNPs after all filters")   
    return snps_df


def select_snps(snps_df, *, snps2keep, n_random, do_pruning, r2_prune_thresh, template_dir, rng_seed):
    # snps2keep: a list of SNP ids to keep
    # n_random: int, number of random SNPs to take
    # do_pruning: True/False - perform pruning
    # r2_prune_thresh, template_dir: required only if do_pruning == True
    print("Selecting variants.")
    b2keep = snps_df.IS_VALID.values.copy()
    print(f"    {b2keep.sum()} remains after applying MAF, Z and INFO filters.")
    rng = np.random.default_rng(seed=rng_seed)
    if snps2keep:
        b2keep &= snps_df.SNP.isin(snps2keep).values
        print(f"    {b2keep.sum()} remains after restricting to provided list of variants.")
    if do_pruning:
        print("    Performing pruning.")
        for chrom in snps_df.CHR.unique():
            ld_r2_file = os.path.join(template_dir, f'chr{chrom}.ld_r2')
            ld_idx_file = os.path.join(template_dir, f'chr{chrom}.ld_idx')
            r2 = np.memmap(ld_r2_file, dtype='f4', mode='r')
            r2_idx = np.memmap(ld_idx_file, dtype='i4', mode='r')
            bchrom = snps_df.CHR == chrom
            rand_pval = rng.random(bchrom.sum())
            ld_n = snps_df.LD_N[bchrom].values
            b2use = b2keep[bchrom]
            bpruned = prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
            print(f"        {bpruned.sum()} SNPs survive pruning on chromosome {chrom}")
            print(f"            {ld_n[bpruned].mean():.2f} mean size of LD block of pruned SNPs")
            b2keep[bchrom] = bpruned
        print(f"    {b2keep.sum()} SNPs remain in total after pruning.")
    if n_random:
        i2keep = np.arange(snps_df.shape[0])[b2keep]
        assert n_random <= len(i2keep)
        i2keep = rng.choice(i2keep, n_random, replace=False) 
        b2keep[:] = False
        b2keep[i2keep] = True
        print(f"    {b2keep.sum()} SNPs remain after taking a random subset.")
    return snps_df.SNP[b2keep].values.copy()
        

def load_opt_data(template_dir, snps_df, *, snps2keep, nbin_het_hist):
    # snps_df is produced by load_snps()
    # snps2keep: a list of SNP ids to keep
    print("Loading LD data")
    z_cols = [c for c in sorted(snps_df.columns) if c.startswith("Z_")]
    n_cols = [c for c in sorted(snps_df.columns) if c.startswith("N_")]
    assert all(z_col.split('_')[1] == n_col.split('_')[1] for z_col, n_col in zip(z_cols, n_cols))
    z_n_dict = {c:[] for c in z_cols + n_cols}
    r2_het_hist_list = []
    ld_scores_list = []
    print("Processing chromosomes: ", end="")
    for chrom in snps_df.CHR.unique():
        print(f"{chrom} ", end="")
        # load template
        ld_r2_file = os.path.join(template_dir, f'chr{chrom}.ld_r2')
        ld_idx_file = os.path.join(template_dir, f'chr{chrom}.ld_idx')
        r2 = np.memmap(ld_r2_file, dtype='f4', mode='r')
        r2_idx = np.memmap(ld_idx_file, dtype='i4', mode='r')
        snps_df_chr = snps_df.loc[snps_df.CHR == chrom,:]
        het = 2*snps_df_chr.MAF.values*(1 - snps_df_chr.MAF.values)
        ld_n = snps_df_chr.LD_N.values
        b2keep = snps_df_chr.SNP.isin(snps2keep).values
        r2_het = get_r2_het(r2, r2_idx, het)
        r2_het_hist = get_r2_het_hist(b2keep, r2_het, ld_n, nbin_het_hist)
        r2_het_hist_list.append(r2_het_hist)
        ld_scores = get_ld_scores(b2keep, r2_het, ld_n, nbin_het_hist)
        ld_scores_list.append(ld_scores)
        
        for z_col, n_col in zip(z_cols, n_cols):
            z = snps_df_chr.loc[b2keep,z_col].values
            n = snps_df_chr.loc[b2keep,n_col].values
            z_n_dict[z_col].append(z)
            z_n_dict[n_col].append(n)
    r2_het_hist = np.concatenate(r2_het_hist_list)
    ld_scores = np.concatenate(ld_scores_list)
    for col, val_list in z_n_dict.items():
        z_n_dict[col] = np.concatenate(val_list)
    print(f"\n{z_n_dict['Z_0'].size} SNPs loaded")
    print(f"{r2_het_hist.sum()/z_n_dict['Z_0'].size:.2f} mean size of LD block of loaded SNPs")
    return r2_het_hist, z_n_dict, ld_scores


# float32 constants
ZERO_f4 = nb.float64(0)
QUARTER_f4 = nb.float64(0.25)
HALF_f4 = nb.float64(0.5)
ONE_f4 = nb.float64(1)
TWO_f4 = nb.float64(2)
PI_f4 = nb.float64(math.pi)

@nb.njit(fastmath=True)
def ch_func_1d(x, p, sb2, s02, n, r2_het_hist):
    nbin_r2_het_hist = r2_het_hist.shape[0]
    nbin_r2_het_hist_inverse = nb.float32(1/nbin_r2_het_hist)
    fx = math.exp(-HALF_f4*x*x*s02)
    for i in range(nbin_r2_het_hist):
        n_in_bin = r2_het_hist[i]
        if n_in_bin != 0:
            rh = (HALF_f4*nb.float32(i) + QUARTER_f4)*nbin_r2_het_hist_inverse # middle of the i-th hist bin with NBIN_R2_HET_HIST bins of [0, 0.5]
            se2 = n*sb2*rh
            fx *= (ONE_f4 - p + p*math.exp(-HALF_f4*x*x*se2))**n_in_bin
    return fx

@nb.njit(fastmath=True)
def ch_func_1d_inductive(x, p, sb2, s02, n, r2_het_hist):
    r2_h = HALF_f4/r2_het_hist.shape[0] # pass as argument
    e_i = math.exp(-QUARTER_f4*x*x*n*sb2*r2_h)
    e_const = e_i**2
    p_null = ONE_f4 - p
    fx = math.exp(-HALF_f4*x*x*s02)
    for n_in_bin in r2_het_hist:
        fx *= (p_null + p*e_i)**n_in_bin
        e_i *= e_const
    return fx


@cuda.jit(fastmath=False) # setting fastmath=True may result in inf numbers and actually reduces speed.
def log_pdf_1d(res_vec, z0_vec, p, sb2, s02, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    # Characteristic function of sum of independent random variables = product of their characteristic functions.
    # Using inversion formula for characteristic function, pdf can be estimated as:
    # pdf = integral{0, inf}( cos(z0*t) * ch_func(t) )dt / pi
    # Integral is calculated using simple trapezoidal rule with fixed step size = h
    # Integration is stoped when ch_func(t) < tol
    # Based on the answer here: https://math.stackexchange.com/questions/2891298/derivation-of-2d-trapezoid-rule
    tid = cuda.grid(1)
    if tid < len(res_vec):
        h = nb.float32(1E-2) # define grid
        min_pdf = nb.float32(1E-30)
        tol = nb.float32(1E-8)
        z0 = z0_vec[tid]
        n = n_vec[tid]
        x = ZERO_f4
        res, ch_fx = ZERO_f4, ONE_f4
        r2_het_hist_tid = r2_het_hist[tid*nbin_r2_het_hist:(tid+1)*nbin_r2_het_hist]
        while ch_fx > tol:
            x = x + h
            ch_fx = ch_func_1d_inductive(x, p, sb2, s02, n, r2_het_hist_tid)
            res += math.cos(z0*x)*ch_fx
        res = ONE_f4 + TWO_f4*res # f(0) = 1
        pdf = HALF_f4*h*res / PI_f4
        res_vec[tid] = -math.log(max(pdf, min_pdf))/ld_scores[tid]

#@cuda.jit(device=True, inline=True)
@nb.njit(fastmath=True)
def ch_func_2d(x, y, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0, n_1, n_2, r2_het_hist):
    nbin_r2_het_hist = r2_het_hist.shape[0]
    nbin_r2_het_hist_inverse = nb.float32(1/nbin_r2_het_hist)
    fx = math.exp(-HALF_f4*(s02_1*x*x + TWO_f4*rho0*math.sqrt(s02_1*s02_2)*x*y + s02_2*y*y))
    for i in range(nbin_r2_het_hist):
        n_in_bin = r2_het_hist[i]
        if n_in_bin != 0:
            rh = (HALF_f4*nb.float32(i) + QUARTER_f4)*nbin_r2_het_hist_inverse
            se2_1 = n_1*sb2_1*rh
            se2_2 = n_2*sb2_2*rh
            fx *= ( ONE_f4 - (p_1+p_2-pp) +
                    (p_1-pp)*math.exp(-HALF_f4*x*x*se2_1) + 
                    (p_2-pp)*math.exp(-HALF_f4*y*y*se2_2) +
                    pp*math.exp(-HALF_f4*(se2_1*x*x + TWO_f4*rho*math.sqrt(se2_1*se2_2)*x*y + se2_2*y*y)) )**n_in_bin
    return fx

@nb.njit(fastmath=True)
def ch_func_2d_inductive(x, y, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0, n_1, n_2, r2_het_hist):
    r2_h = HALF_f4/r2_het_hist.shape[0] # pass as argument
    e1_i = math.exp(-QUARTER_f4*x*x*n_1*sb2_1*r2_h)
    e2_i = math.exp(-QUARTER_f4*y*y*n_2*sb2_2*r2_h)
    e12_i = math.exp(-HALF_f4*rho*x*y*r2_h*math.sqrt(n_1*sb2_1*n_2*sb2_2))
    e1_const = e1_i**2
    e2_const = e2_i**2
    e12_const = e12_i**2
    p_null = ONE_f4 - p_1 - p_2 + pp
    p1_only = p_1 - pp
    p2_only = p_2 - pp
    fx = math.exp(-HALF_f4*(s02_1*x*x + TWO_f4*rho0*math.sqrt(s02_1*s02_2)*x*y + s02_2*y*y))
    for n_in_bin in r2_het_hist:
        fx *= (p_null + p1_only*e1_i + p2_only*e2_i + pp*e1_i*e2_i*e12_i)**n_in_bin
        e1_i *= e1_const
        e2_i *= e2_const
        e12_i *= e12_const
    return fx
        

@cuda.jit(fastmath=False)
def log_pdf_2d(res_vec, z0_1_vec, z0_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0,
               n_1_vec, n_2_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    tid = cuda.grid(1)
    if tid < len(res_vec):
        h = nb.float32(5E-2) # define grid
        min_pdf = nb.float32(1E-30)
        tol = nb.float32(1E-6)
        z0_1, z0_2 = z0_1_vec[tid], z0_2_vec[tid]
        n_1, n_2 = n_1_vec[tid], n_2_vec[tid]
        x = ZERO_f4 # start from x == 0 and go up x axis
        factor = TWO_f4 # factor = 2 if x == 0 else 4
        ch_fx_max = tol + ONE_f4 # arbitrary value > tol
        y_ch_fx_max = ZERO_f4 # start with 0, then y_ch_fx_max: ch_func_2d(x,y_ch_fx_max) == ch_fx_max for current x
        pdf = ZERO_f4
        r2_het_hist_tid = r2_het_hist[tid*nbin_r2_het_hist:(tid+1)*nbin_r2_het_hist]
        while ch_fx_max > tol:
            ch_fx_max = tol
            # go up y axis from y = y_ch_fx_max
            y_start = y_ch_fx_max
            y = y_start
            res = ZERO_f4 # accumulates sum of integrated func over y for a fixed x.
            ch_fx = tol + ONE_f4 # arbitrary value > tol
            while ch_fx > tol:
                ch_fx = ch_func_2d_inductive(x, y, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0,
                                   n_1, n_2, r2_het_hist_tid)
                if ch_fx_max < ch_fx:
                    ch_fx_max = ch_fx
                    y_ch_fx_max = y
                res += math.cos(z0_1*x + z0_2*y)*ch_fx
                y += h
            # go down y axis from y = y_start - h
            y = y_start - h
            ch_fx = tol + ONE_f4 # arbitrary value > tol
            while ch_fx > tol:
                ch_fx = ch_func_2d_inductive(x, y, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0,
                                   n_1, n_2, r2_het_hist_tid)
                if ch_fx_max < ch_fx:
                    ch_fx_max = ch_fx
                    y_ch_fx_max = y
                res += math.cos(z0_1*x + z0_2*y)*ch_fx
                y -= h
            pdf += factor*res
            x += h
            factor = TWO_f4*TWO_f4
        pdf *= HALF_f4*h*h / (TWO_f4*PI_f4)**TWO_f4
        res_vec[tid] = -math.log(max(pdf, min_pdf))/ld_scores[tid]


@nb.njit(fastmath=True)
def ch_func_3d(x, y, z, n_1, n_2, n_3,
               p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
               p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23,
               p_123, r2_het_hist):
    nbin_r2_het_hist = r2_het_hist.shape[0]
    nbin_r2_het_hist_inverse = nb.float32(1/nbin_r2_het_hist)
    p_12_only = p_12 - p_123 
    p_13_only = p_13 - p_123 
    p_23_only = p_23 - p_123
    p_1_only = p_1 - p_12_only - p_13_only - p_123
    p_2_only = p_2 - p_12_only - p_23_only - p_123
    p_3_only = p_3 - p_13_only - p_23_only - p_123
    p_null = ONE_f4 - p_123 - p_12_only - p_13_only - p_23_only - p_1_only - p_2_only - p_3_only
    fx = math.exp(-HALF_f4*(s02_1*x*x + s02_2*y*y + s02_3*z*z + rho0_12*math.sqrt(s02_1*s02_2)*x*y + rho0_13*math.sqrt(s02_1*s02_3)*x*z + rho0_23*math.sqrt(s02_2*s02_3)*y*z))
    for i in range(nbin_r2_het_hist):
        n_in_bin = r2_het_hist[i]
        if n_in_bin != 0:
            rh = (HALF_f4*nb.float32(i) + QUARTER_f4)*nbin_r2_het_hist_inverse
            se2_1 = n_1*sb2_1*rh
            se2_2 = n_2*sb2_2*rh
            se2_3 = n_3*sb2_3*rh
            cxx = se2_1*x*x
            cyy = se2_2*y*y
            czz = se2_3*z*z
            cxy = TWO_f4*rho_12*math.sqrt(se2_1*se2_2)*x*y
            cxz = TWO_f4*rho_13*math.sqrt(se2_1*se2_3)*x*z
            cyz = TWO_f4*rho_23*math.sqrt(se2_2*se2_3)*y*z
            fx *= ( p_null +
                    p_1_only*math.exp(-HALF_f4*cxx) + 
                    p_2_only*math.exp(-HALF_f4*cyy) +
                    p_3_only*math.exp(-HALF_f4*czz) +
                    p_12_only*math.exp(-HALF_f4*(cxx + cxy + cyy)) +
                    p_13_only*math.exp(-HALF_f4*(cxx + cxz + czz)) +
                    p_23_only*math.exp(-HALF_f4*(cyy + cyz + czz)) +
                    p_123*math.exp(-HALF_f4*(cxx + cyy + czz + cxy + cxz + cyz))
                  )**n_in_bin
    return fx

@nb.njit(fastmath=True)    
def ch_func_3d_inductive(x, y, z, n_1, n_2, n_3,    
               p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,    
               p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23,    
               p_123, r2_het_hist):
    r2_h = HALF_f4/r2_het_hist.shape[0] # pass as argument
    e1_i = math.exp(-QUARTER_f4*x*x*n_1*sb2_1*r2_h)
    e2_i = math.exp(-QUARTER_f4*y*y*n_2*sb2_2*r2_h)
    e3_i = math.exp(-QUARTER_f4*z*z*n_3*sb2_3*r2_h)
    e12_i = math.exp(-HALF_f4*rho_12*x*y*r2_h*math.sqrt(n_1*sb2_1*n_2*sb2_2))
    e13_i = math.exp(-HALF_f4*rho_13*x*z*r2_h*math.sqrt(n_1*sb2_1*n_3*sb2_3))
    e23_i = math.exp(-HALF_f4*rho_23*y*z*r2_h*math.sqrt(n_2*sb2_2*n_3*sb2_3))
    e1_const = e1_i**2
    e2_const = e2_i**2
    e3_const = e3_i**2
    e12_const = e12_i**2
    e13_const = e13_i**2
    e23_const = e23_i**2
    p_12_only = p_12 - p_123     
    p_13_only = p_13 - p_123     
    p_23_only = p_23 - p_123    
    p_1_only = p_1 - p_12_only - p_13_only - p_123    
    p_2_only = p_2 - p_12_only - p_23_only - p_123    
    p_3_only = p_3 - p_13_only - p_23_only - p_123    
    p_null = ONE_f4 - p_123 - p_12_only - p_13_only - p_23_only - p_1_only - p_2_only - p_3_only    
    fx = math.exp(-HALF_f4*(s02_1*x*x + s02_2*y*y + s02_3*z*z + rho0_12*math.sqrt(s02_1*s02_2)*x*y + rho0_13*math.sqrt(s02_1*s02_3)*x*z + rho0_23*math.sqrt(s02_2*s02_3)*y*z))
    for n_in_bin in r2_het_hist:
        fx *= (p_null + p_1_only*e1_i + p_2_only*e2_i + p_3_only*e3_i +
               p_12_only*e1_i*e2_i*e12_i + p_13_only*e1_i*e3_i*e13_i + p_23_only*e2_i*e3_i*e23_i +
               p_123*e1_i*e2_i*e3_i*e12_i*e13_i*e23_i)**n_in_bin
        e1_i *= e1_const
        e2_i *= e2_const
        e3_i *= e3_const
        e12_i *= e12_const
        e13_i *= e13_const
        e23_i *= e23_const
    return fx

@cuda.jit(fastmath=False)
def log_pdf_3d(res_vec, z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
               p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
               p_123, r2_het_hist, ld_scores, nbin_r2_het_hist):
    tid = cuda.grid(1)
    if tid < len(res_vec):
        h = nb.float32(1E-1) # define grid
        min_pdf = nb.float32(1E-30)
        tol = nb.float32(1E-6)
        z0_1, z0_2, z0_3 = z0_1_vec[tid], z0_2_vec[tid], z0_3_vec[tid]
        n_1, n_2, n_3 = n_1_vec[tid], n_2_vec[tid], n_3_vec[tid]
        r2_het_hist_tid = r2_het_hist[tid*nbin_r2_het_hist:(tid+1)*nbin_r2_het_hist]
        ch_func_args = (n_1, n_2, n_3, p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23, p_123, r2_het_hist_tid)

        pdf = ZERO_f4
        x = ZERO_f4 # start from x == 0 and go up x axis
        factor = TWO_f4*TWO_f4 # factor = 4 if x == 0 else 8
        fx_max_yz_prev = tol + ONE_f4 
        y_max_yz_prev = ZERO_f4
        z_max_yz_prev = ZERO_f4
        while fx_max_yz_prev > tol:
            fx_max_yz_cur = ZERO_f4
            y_max_yz_cur = ZERO_f4
            z_max_yz_cur = ZERO_f4
            res = ZERO_f4 # accumulates sum of integrated func over y for a fixed x.
            y_start = y_max_yz_prev
            fx_max_z_prev = ONE_f4 + tol
            z_max_z_prev = z_max_yz_prev
            # go up y
            y = y_start
            while fx_max_z_prev > tol:
                fx_max_z_cur = 0
                z_max_z_cur = 0

                z_start = z_max_z_prev
                z = z_start
                fx = ONE_f4 + tol
                # go up z
                while fx > tol:
                    fx = ch_func_3d_inductive(x, y, z, *ch_func_args)
                    res += math.cos(z0_1*x + z0_2*y + z0_3*z)*fx
                    if fx > fx_max_z_cur:
                        fx_max_z_cur = fx
                        z_max_z_cur = z
                    z += h
                # go down z
                z = z_start - h
                fx = tol + ONE_f4 # arbitrary value > tol
                while fx > tol:
                    fx = ch_func_3d_inductive(x, y, z, *ch_func_args)
                    res += math.cos(z0_1*x + z0_2*y + z0_3*z)*fx
                    if fx > fx_max_z_cur:
                        fx_max_z_cur = fx
                        z_max_z_cur = z
                    z -= h

                fx_max_z_prev = fx_max_z_cur
                z_max_z_prev = z_max_z_cur

                if fx_max_z_cur > fx_max_yz_cur:
                    fx_max_yz_cur = fx_max_z_cur
                    y_max_yz_cur = y
                    z_max_yz_cur = z_max_z_cur

                y += h
            # go down y
            fx_max_z_prev = ONE_f4 + tol
            z_max_z_prev = z_max_yz_prev
            y = y_start - h
            while fx_max_z_prev > tol:
                fx_max_z_cur = ZERO_f4
                z_max_z_cur = ZERO_f4

                z_start = z_max_z_prev
                z = z_start
                fx = ONE_f4 + tol
                # go up z
                while fx > tol:
                    fx = ch_func_3d_inductive(x, y, z, *ch_func_args)
                    res += math.cos(z0_1*x + z0_2*y + z0_3*z)*fx
                    if fx > fx_max_z_cur:
                        fx_max_z_cur = fx
                        z_max_z_cur = z
                    z += h
                z = z_start - h
                fx = ONE_f4 + tol
                # go down z
                while fx > tol:
                    fx = ch_func_3d_inductive(x, y, z, *ch_func_args)
                    res += math.cos(z0_1*x + z0_2*y + z0_3*z)*fx
                    if fx > fx_max_z_cur:
                        fx_max_z_cur = fx
                        z_max_z_cur = z
                    z -= h

                fx_max_z_prev = fx_max_z_cur
                z_max_z_prev = z_max_z_cur

                if fx_max_z_cur > fx_max_yz_cur:
                    fx_max_yz_cur = fx_max_z_cur
                    y_max_yz_cur = y
                    z_max_yz_cur = z_max_z_cur

                y -= h

            fx_max_yz_prev = fx_max_yz_cur
            y_max_yz_prev = y_max_yz_cur
            z_max_yz_prev = z_max_yz_cur
            x += h
            pdf += factor*res
            factor = TWO_f4*TWO_f4*TWO_f4
        
        # integral in the x > 0 half-space is equal to the integral in x < 0 subspace
        pdf *= QUARTER_f4*h*h*h / (TWO_f4*PI_f4)**(TWO_f4 + ONE_f4)
        res_vec[tid] = -math.log(max(pdf, min_pdf)) / ld_scores[tid]


@cuda.reduce
def sum_reduce(a, b):
    return a + b


def cost_1d_gpu(z0_vec, p, sb2, s02, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    p, sb2, s02 = map(nb.float32, (p, sb2, s02)) 
    log_pdf_vec_gpu = cuda.device_array_like(z0_vec)
    r2_het_hist_gpu = cuda.to_device(r2_het_hist)
    z0_vec_gpu = cuda.to_device(z0_vec)
    n_vec_gpu = cuda.to_device(n_vec)
    ld_scores_gpu = cuda.to_device(ld_scores)
    
    log_pdf_1d.forall(len(z0_vec))(log_pdf_vec_gpu, z0_vec_gpu, p, sb2, s02, n_vec_gpu,
                                   r2_het_hist_gpu, ld_scores_gpu, nbin_r2_het_hist)
    cost = sum_reduce(log_pdf_vec_gpu) 
    return cost

def cost_2d_gpu(z0_1_vec, z0_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0, n_1_vec, n_2_vec,
                    r2_het_hist, ld_scores, nbin_r2_het_hist):
    p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0 = map(nb.float32,
                                                              (p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0))
    log_pdf_vec_gpu = cuda.device_array_like(z0_1_vec)
    
    r2_het_hist_gpu = cuda.to_device(r2_het_hist)
    z0_1_vec_gpu = cuda.to_device(z0_1_vec)
    z0_2_vec_gpu = cuda.to_device(z0_2_vec)
    n_1_vec_gpu = cuda.to_device(n_1_vec)
    n_2_vec_gpu = cuda.to_device(n_2_vec)
    ld_scores_gpu = cuda.to_device(ld_scores)
    
    log_pdf_2d.forall(len(z0_1_vec))(log_pdf_vec_gpu,
                                     z0_1_vec_gpu, z0_2_vec_gpu, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2,
                                     pp, rho, rho0, n_1_vec_gpu, n_2_vec_gpu, r2_het_hist_gpu, ld_scores_gpu, nbin_r2_het_hist)
    cost = sum_reduce(log_pdf_vec_gpu)
    return cost

def cost_3d_gpu(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                p_123, r2_het_hist, ld_scores, nbin_r2_het_hist):
    p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23, p_123 = map(nb.float32,
                                                                                                                                               (p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                                                                                                                                                p_12, p_13, p_23, rho_12, rho_13, rho_23,
                                                                                                                                                rho0_12, rho0_13, rho0_23, p_123))
    log_pdf_vec_gpu = cuda.device_array_like(z0_1_vec)
    
    r2_het_hist_gpu = cuda.to_device(r2_het_hist)
    z0_1_vec_gpu = cuda.to_device(z0_1_vec)
    z0_2_vec_gpu = cuda.to_device(z0_2_vec)
    z0_3_vec_gpu = cuda.to_device(z0_3_vec)
    n_1_vec_gpu = cuda.to_device(n_1_vec)
    n_2_vec_gpu = cuda.to_device(n_2_vec)
    n_3_vec_gpu = cuda.to_device(n_3_vec)
    ld_scores_gpu = cuda.to_device(ld_scores)
    
    log_pdf_3d.forall(len(z0_1_vec))(log_pdf_vec_gpu,
                                     z0_1_vec_gpu, z0_2_vec_gpu, z0_3_vec_gpu, n_1_vec_gpu, n_2_vec_gpu, n_3_vec_gpu,
                                     p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3, p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23,
                                     p_123, r2_het_hist_gpu, ld_scores_gpu, nbin_r2_het_hist)
    cost = sum_reduce(log_pdf_vec_gpu)
    return cost


def obj_func_1d(par_vec, z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist):
    p, sb2, s02 = par_vec
    p = 10**p
    sb2 = 10**sb2
    cost = cost_1d_gpu(z0_vec, p, sb2, s02, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    print(f"cost = {cost:.7f}, p = {p:.7e}, sb2 = {sb2:.7e}, s02 = {s02:.7f}", flush=True)
    return cost


def optimize_1d(z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist,
                z0_vec_global=None, n_vec_global=None, r2_het_hist_global=None, ld_scores_global=None,
                maxiter_1d_glob=3200, maxiter_1d_loc=100):
    p_lb, p_rb = -6, -1 # on log10 scale
    sb2_lb, sb2_rb = -7, -2 # on log10 scale
    s02_lb, s02_rb = 0.8, 2.5

    bounds = [(p_lb, p_rb), (sb2_lb, sb2_rb), (s02_lb, s02_rb)]
    args_opt = (z0_vec, n_vec, r2_het_hist, ld_scores, nbin_r2_het_hist)
    
    if z0_vec_global is None:
        args_opt_global = args_opt
    else:
        args_opt_global = (z0_vec_global, n_vec_global, r2_het_hist_global, ld_scores_global, nbin_r2_het_hist)
        
    print(">>> Starting global optimization. ------------------------------------------------------")
    res = direct(obj_func_1d, bounds, args=args_opt_global, maxfun=maxiter_1d_glob, locally_biased=True)
    
    print(">>> Starting local optimization. -------------------------------------------------------")
    x0 = res.x
    res = minimize(obj_func_1d, x0=x0, args=args_opt, method='Nelder-Mead', bounds=bounds,
            options={'maxfev':maxiter_1d_loc, 'fatol':1, 'xatol':1E-3, 'adaptive':True})
    
    opt_par = [10**res.x[0], 10**res.x[1], res.x[2]]
    opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
    for k in ("success", "status", "message", "nfev", "nit"):
        opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par)
    return opt_out

def obj_func_2d(par_vec, z0_1_vec, z0_2_vec, n_1_vec, n_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, r2_het_hist, ld_scores, nbin_r2_het_hist):
    pp, rho, rho0 = par_vec
    pp = 10**pp
    
    cost = cost_2d_gpu(z0_1_vec, z0_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, pp, rho, rho0, n_1_vec, n_2_vec,
            r2_het_hist, ld_scores, nbin_r2_het_hist)
    print(f"cost = {cost:.7f}, p12 = {pp:.7e}, rho = {rho:.7f}, rho0 = {rho0:.7f}", flush=True)
    return cost

def optimize_2d(p_1, sb2_1, s02_1, n_1_vec, z0_1_vec, p_2, sb2_2, s02_2, n_2_vec,
                z0_2_vec, r2_het_hist, ld_scores, nbin_r2_het_hist,
                z0_1_vec_global=None, n_1_vec_global=None, z0_2_vec_global=None, n_2_vec_global=None,
                r2_het_hist_global=None, ld_scores_global=None, maxiter_2d_glob=3200, maxiter_2d_loc=100):
    p12_lb, p12_rb = -6, np.log10(min(p_1,p_2)) # on log10 scale
    assert p12_lb < p12_rb
    rho_lb, rho_rb = -1, 1
    rho0_lb, rho0_rb = -1, 1
     
    bounds = [(p12_lb, p12_rb), (rho_lb, rho_rb), (rho0_lb, rho0_rb)]
    args_opt = (z0_1_vec, z0_2_vec, n_1_vec, n_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, r2_het_hist, ld_scores, nbin_r2_het_hist)
    
    if z0_1_vec_global is None:
        args_opt_global = args_opt
    else:
        args_opt_global = (z0_1_vec_global, z0_2_vec_global, n_1_vec_global, n_2_vec_global, p_1, p_2,
                           sb2_1, sb2_2, s02_1, s02_2, r2_het_hist_global, ld_scores_global, nbin_r2_het_hist)
    
    print(">>> Starting global optimization. ------------------------------------------------------")
    res = direct(obj_func_2d, bounds, args=args_opt_global, maxfun=maxiter_2d_glob, locally_biased=True)
    
    print(">>> Starting local optimization. -------------------------------------------------------")
    x0 = res.x
    res = minimize(obj_func_2d, x0=x0, args=args_opt, method='Nelder-Mead', bounds=bounds,
            options={'maxfev':maxiter_2d_loc, 'fatol':1, 'xatol':1E-3, 'adaptive':True})
    
    opt_par = [10**res.x[0], res.x[1], res.x[2]]
    opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
    for k in ("success", "status", "message", "nfev", "nit"):
        opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par)
    return opt_out


def obj_func_2d_constr(par_vec, p_1, sb2_1, s02_1, p_2, sb2_2, s02_2, p_3, sb2_3, s02_3,
                   z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec, r2_het_hist, ld_scores, nbin_het_hist):
    p_12 = 10**par_vec[0]
    p_13 = 10**par_vec[1]
    p_23 = 10**par_vec[2]
    rho_12 = par_vec[3]
    rho_13 = par_vec[4]
    rho_23 = par_vec[5]
    rho0_12 = par_vec[6]
    rho0_13 = par_vec[7]
    rho0_23 = par_vec[8]
    
    cost_12 = cost_2d_gpu(z0_1_vec, z0_2_vec, p_1, p_2, sb2_1, sb2_2, s02_1, s02_2, p_12, rho_12, rho0_12, n_1_vec, n_2_vec,
            r2_het_hist, ld_scores, nbin_het_hist)
    cost_13 = cost_2d_gpu(z0_1_vec, z0_3_vec, p_1, p_3, sb2_1, sb2_3, s02_1, s02_3, p_13, rho_13, rho0_13, n_1_vec, n_3_vec,
            r2_het_hist, ld_scores, nbin_het_hist)
    cost_23 = cost_2d_gpu(z0_2_vec, z0_3_vec, p_2, p_3, sb2_2, sb2_3, s02_2, s02_3, p_23, rho_23, rho0_23, n_2_vec, n_3_vec,
            r2_het_hist, ld_scores, nbin_het_hist)
    cost = cost_12 + cost_13 + cost_23
    print(f"cost = {cost:.7f}")
    print(f"    cost12 = {cost_12:.7f}, p12 = {p_12:.7e}, rho12 = {rho_12:.7e}, rho012 = {rho0_12:.7e}")
    print(f"    cost13 = {cost_13:.7f}, p13 = {p_13:.7e}, rho13 = {rho_13:.7e}, rho013 = {rho0_13:.7e}")
    print(f"    cost23 = {cost_23:.7f}, p23 = {p_23:.7e}, rho23 = {rho_23:.7e}, rho023 = {rho0_23:.7e}", flush=True)
    return cost


def push_to_bounds(val, lb, rb):
    if val < lb:
        val = lb
    elif val > rb:
        val = rb
    return val
    
def optimize_2d_constr(p_1, sb2_1, s02_1, p_2, sb2_2, s02_2, p_3, sb2_3, s02_3,
                       p_12, rho_12, rho0_12, p_13, rho_13, rho0_13, p_23, rho_23, rho0_23,
                       z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                       r2_het_hist, ld_scores, nbin_het_hist):
    max_rel_dev_p = 0.1
    max_rel_dev_rho = 0.12
    max_rel_dev_rho0 = 0.05
    p_12_lb, p_12_rb = (1+max_rel_dev_p)*math.log10(p_12), (1-max_rel_dev_p)*math.log10(p_12) # on log10 scale    0 
    p_12_lb = push_to_bounds(p_12_lb, -6, -1)
    p_12_rb = push_to_bounds(p_12_rb, -6, -1)
    p_13_lb, p_13_rb = (1+max_rel_dev_p)*math.log10(p_13), (1-max_rel_dev_p)*math.log10(p_13) # on log10 scale    0 
    p_13_lb = push_to_bounds(p_13_lb, -6, -1)
    p_13_rb = push_to_bounds(p_13_rb, -6, -1)
    p_23_lb, p_23_rb = (1+max_rel_dev_p)*math.log10(p_23), (1-max_rel_dev_p)*math.log10(p_23) # on log10 scale    0 
    p_23_lb = push_to_bounds(p_23_lb, -6, -1)
    p_23_rb = push_to_bounds(p_23_rb, -6, -1)
    rho_12_lb, rho_12_rb = (1-max_rel_dev_rho)*rho_12, (1+max_rel_dev_rho)*rho_12
    rho_12_lb = push_to_bounds(rho_12_lb, -1, 1)
    rho_12_rb = push_to_bounds(rho_12_rb, -1, 1)
    rho_13_lb, rho_13_rb = (1-max_rel_dev_rho)*rho_13, (1+max_rel_dev_rho)*rho_13
    rho_13_lb = push_to_bounds(rho_13_lb, -1, 1)
    rho_13_rb = push_to_bounds(rho_13_rb, -1, 1)
    rho_23_lb, rho_23_rb = (1-max_rel_dev_rho)*rho_23, (1+max_rel_dev_rho)*rho_23
    rho_23_lb = push_to_bounds(rho_23_lb, -1, 1)
    rho_23_rb = push_to_bounds(rho_23_rb, -1, 1)
    rho0_12_lb, rho0_12_rb = (1-max_rel_dev_rho0)*rho0_12, (1+max_rel_dev_rho0)*rho0_12
    rho0_12_lb = push_to_bounds(rho0_12_lb, -1, 1)
    rho0_12_rb = push_to_bounds(rho0_12_rb, -1, 1)
    rho0_13_lb, rho0_13_rb = (1-max_rel_dev_rho0)*rho0_13, (1+max_rel_dev_rho0)*rho0_13
    rho0_13_lb = push_to_bounds(rho0_13_lb, -1, 1)
    rho0_13_rb = push_to_bounds(rho0_13_rb, -1, 1)
    rho0_23_lb, rho0_23_rb = (1-max_rel_dev_rho0)*rho0_23, (1+max_rel_dev_rho0)*rho0_23
    rho0_23_lb = push_to_bounds(rho0_23_lb, -1, 1)
    rho0_23_rb = push_to_bounds(rho0_23_rb, -1, 1)
    bounds = [(p_12_lb,p_12_rb), (p_13_lb,p_13_rb), (p_23_lb,p_23_rb),
              (rho_12_lb, rho_12_rb), (rho_13_lb, rho_13_rb), (rho_23_lb, rho_23_rb),
              (rho0_12_lb, rho0_12_rb), (rho0_13_lb, rho0_13_rb), (rho0_23_lb, rho0_23_rb)]

    con_fun_1 = lambda x: p_1 - 10**x[0] - 10**x[1] + min(10**x[0],10**x[1],10**x[2])
    con_1 = NonlinearConstraint(con_fun_1, 0, np.inf)
    con_fun_2 = lambda x: p_2 - 10**x[0] - 10**x[2] + min(10**x[0],10**x[1],10**x[2])
    con_2 = NonlinearConstraint(con_fun_2, 0, np.inf)
    con_fun_3 = lambda x: p_3 - 10**x[1] - 10**x[2] + min(10**x[0],10**x[1],10**x[2])
    con_3 = NonlinearConstraint(con_fun_3, 0, np.inf)
    constraints = (con_1, con_2, con_3)
    
    args_opt = (p_1, sb2_1, s02_1, p_2, sb2_2, s02_2, p_3, sb2_3, s02_3,
                z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec, r2_het_hist, ld_scores, nbin_het_hist)

    print(">>> Starting global optimization. ------------------------------------------------------")
    print("Parameter bounds:")
    print(bounds)
    res = differential_evolution(obj_func_2d_constr, bounds, args=args_opt, popsize=10,
                                 polish=False, constraints=constraints, maxiter=4)
    opt_par = [10**res.x[0], 10**res.x[1], 10**res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]]
    opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
    for k in ("success", "status", "message", "nfev", "nit"):
        opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par)
    return opt_out


def obj_func_3d(p_123, z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                r2_het_hist, ld_scores, nbin_het_hist):
    p_123 = 10**p_123
    
    cost = cost_3d_gpu(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                       p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                       p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                       p_123, r2_het_hist, ld_scores, nbin_het_hist)

    print(f"cost = {cost:.7f}, p123 = {p_123:.7e}", flush=True)
    return cost

def obj_func_3d_brute(p_123_vec, z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                r2_het_hist, ld_scores, nbin_het_hist):
    p_123 = 10**p_123_vec[0]
    
    cost = cost_3d_gpu(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                       p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                       p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                       p_123, r2_het_hist, ld_scores, nbin_het_hist)

    print(f"cost = {cost:.7f}, p123 = {p_123:.7e}", flush=True)
    return cost

def optimize_3d(z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                r2_het_hist, ld_scores, nbin_het_hist, maxiter):
    p_123_lb, p_123_rb = math.log10(max(1E-6, p_12+p_13-p_1, p_12+p_23-p_2, p_13+p_23-p_3)), math.log10(min(p_12, p_13, p_23)) # on log10 scale
    print(10**p_123_lb, 10**p_123_rb)
    if p_123_lb >= p_123_rb:
        print("Trivariate optimization is skipped")
        opt_par = [10**p_123_rb]
        opt_res = {"x":[p_123_rb], "fun":[None] ,"success":False, "status":1,
                   "message":"Trivariate optimization is skipped",
                   "nfev":0, "nit":0}
    else:
        print(">>> Starting global optimization. -------------------------------------------------------")
        args_opt = (z0_1_vec, z0_2_vec, z0_3_vec, n_1_vec, n_2_vec, n_3_vec,
                    p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                    p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                    r2_het_hist, ld_scores, nbin_het_hist)
        if False:
            res = minimize_scalar(obj_func_3d, args=args_opt, method='bounded',
                                  bounds=(p_123_lb, p_123_rb), options={'maxiter':maxiter, 'xatol':1E-4})
            opt_par = [10**res.x]
            opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
            for k in ("success", "status", "message", "nfev", "nit"):
                opt_res[k] = res.get(k)
        else:
            print("Using brute optimization.")
            x0, fval, grid, Jout = brute(obj_func_3d_brute, ranges=((p_123_lb, p_123_rb),), args=args_opt, Ns=maxiter, finish=False, full_output=True)
            x0, fval = float(x0), float(fval) # to make it JSON serializable
            opt_par = [10**x0]
            opt_res = dict(x=[x0], fun=[fval])
            opt_res["success"] = True
            opt_res["status"] = None
            opt_res["message"] = ""
            opt_res["nfev"] = maxiter
            opt_res["nit"] = maxiter
    opt_out = dict(opt_res=opt_res, opt_par=opt_par)
    return opt_out

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    nbin_het_hist = config["nbin_het_hist"]
    print(f"{nbin_het_hist} bins in het hist.")

    snps_df = load_snps(config["template_dir"], config["sumstats"],
            chromosomes=config["snp_filters"]["chromosomes"],
            z_thresh=config["snp_filters"]["z_thresh"],
            info_thresh=config["snp_filters"]["info_thresh"],
            maf_thresh=config["snp_filters"]["maf_thresh"],
            exclude_regions=config["snp_filters"]["exclude_regions"])

    snps2keep = select_snps(snps_df, snps2keep=None, n_random=config["pruning"]["n_random"], do_pruning=config["pruning"]["do_pruning"],
                        r2_prune_thresh=config["pruning"]["r2_prune_thresh"],
                        template_dir=config["template_dir"],
                        rng_seed=config["pruning"]["rand_prune_seed"])

    r2_het_hist, z_n_dict, ld_scores = load_opt_data(config["template_dir"], snps_df,
                                      snps2keep=snps2keep, nbin_het_hist=nbin_het_hist)

    r2_het_hist_global, z_n_dict_global, ld_scores_global = r2_het_hist, z_n_dict, ld_scores

    if True:
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_1 = optimize_1d(z_n_dict["Z_0"], z_n_dict["N_0"], r2_het_hist, ld_scores, nbin_het_hist,
                               z_n_dict_global["Z_0"], z_n_dict_global["N_0"], r2_het_hist_global, ld_scores_global,
                                maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
                                maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"])

        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 1:")
        print(opt_out_1)


    if True:
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_2 = optimize_1d(z_n_dict["Z_1"], z_n_dict["N_1"], r2_het_hist, ld_scores, nbin_het_hist,
                               z_n_dict_global["Z_1"], z_n_dict_global["N_1"], r2_het_hist_global, ld_scores_global,
                                maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
                                maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 2:")
        print(opt_out_2)

    if True:
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_3 = optimize_1d(z_n_dict["Z_2"], z_n_dict["N_2"], r2_het_hist, ld_scores, nbin_het_hist,
                                z_n_dict_global["Z_2"], z_n_dict_global["N_2"], r2_het_hist_global, ld_scores_global,
                                maxiter_1d_glob=config["optimization"]["maxiter_1d_glob"],
                                maxiter_1d_loc=config["optimization"]["maxiter_1d_loc"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Univariate result 3:")
        print(opt_out_3)


    if True:
        p_1, sb2_1, s02_1 = opt_out_1['opt_par']
        p_2, sb2_2, s02_2 = opt_out_2['opt_par']
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_12 = optimize_2d(p_1, sb2_1, s02_1, z_n_dict["N_0"], z_n_dict["Z_0"], p_2, sb2_2, s02_2,
                                 z_n_dict["N_1"], z_n_dict["Z_1"],
                                 r2_het_hist, ld_scores, nbin_het_hist,
                                 z_n_dict_global["Z_0"], z_n_dict_global["N_0"],
                                 z_n_dict_global["Z_1"], z_n_dict_global["N_1"], r2_het_hist_global, ld_scores_global,
                                 maxiter_2d_glob=config["optimization"]["maxiter_2d_glob"],
                                 maxiter_2d_loc=config["optimization"]["maxiter_2d_loc"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 1 vs 2:")
        print(opt_out_12)

    if True:
        p_1, sb2_1, s02_1 = opt_out_1['opt_par']
        p_3, sb2_3, s02_3 = opt_out_3['opt_par']
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_13 = optimize_2d(p_1, sb2_1, s02_1, z_n_dict["N_0"], z_n_dict["Z_0"], p_3, sb2_3, s02_3,
                                 z_n_dict["N_2"], z_n_dict["Z_2"],
                                 r2_het_hist, ld_scores, nbin_het_hist,
                                 z_n_dict_global["Z_0"], z_n_dict_global["N_0"],
                                 z_n_dict_global["Z_2"], z_n_dict_global["N_2"], r2_het_hist_global, ld_scores_global,
                                 maxiter_2d_glob=config["optimization"]["maxiter_2d_glob"],
                                 maxiter_2d_loc=config["optimization"]["maxiter_2d_loc"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 1 vs 3:")
        print(opt_out_13)

    if True:
        p_2, sb2_2, s02_2 = opt_out_2['opt_par']
        p_3, sb2_3, s02_3 = opt_out_3['opt_par']
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_23 = optimize_2d(p_2, sb2_2, s02_2, z_n_dict["N_1"], z_n_dict["Z_1"], p_3, sb2_3, s02_3,
                                 z_n_dict["N_2"], z_n_dict["Z_2"],
                                 r2_het_hist, ld_scores, nbin_het_hist,
                                 z_n_dict_global["Z_1"], z_n_dict_global["N_1"],
                                 z_n_dict_global["Z_2"], z_n_dict_global["N_2"], r2_het_hist_global, ld_scores_global,
                                 maxiter_2d_glob=config["optimization"]["maxiter_2d_glob"],
                                 maxiter_2d_loc=config["optimization"]["maxiter_2d_loc"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Bivariate result 2 vs 3:")
        print(opt_out_23)

    if True:
        p_1, sb2_1, s02_1 = opt_out_1['opt_par']
        p_2, sb2_2, s02_2 = opt_out_2['opt_par']
        p_3, sb2_3, s02_3 = opt_out_3['opt_par']
        p_12, rho_12, rho0_12 = opt_out_12['opt_par']
        p_13, rho_13, rho0_13 = opt_out_13['opt_par']
        p_23, rho_23, rho0_23 = opt_out_23['opt_par']

        p_123_lb, p_123_rb = math.log10(max(1E-6, p_12+p_13-p_1, p_12+p_23-p_2, p_13+p_23-p_3)), math.log10(min(p_12, p_13, p_23)) # on log10 scale
        if p_123_lb > p_123_rb:
            print("Run triple bivariate analysis to make parameters feasible for trivariate.")
            now = datetime.now()
            start_time = now.strftime("%D-%H:%M:%S")
            opt_out_12_13_23 = optimize_2d_constr(p_1, sb2_1, s02_1, p_2, sb2_2, s02_2, p_3, sb2_3, s02_3,
                                                   p_12, rho_12, rho0_12, p_13, rho_13, rho0_13, p_23, rho_23, rho0_23,
                                                   z_n_dict["Z_0"], z_n_dict["Z_1"], z_n_dict["Z_2"], z_n_dict["N_0"], z_n_dict["N_1"], z_n_dict["N_2"],
                                                   r2_het_hist, ld_scores, nbin_het_hist)
            now = datetime.now()
            end_time = now.strftime("%D-%H:%M:%S")
            print("Start Time =", start_time)
            print("End Time =", end_time)
            print("Bivariate constrained result 1 vs 2 vs 3:")
            print(opt_out_12_13_23)
            p_12, p_13, p_23, rho_12, rho_13, rho_23, rho0_12, rho0_13, rho0_23 = opt_out_12_13_23['opt_par'] # reassign bivariate parameters
        else:
            opt_out_12_13_23 = None
        # Run trivariate analysis
        now = datetime.now()
        start_time = now.strftime("%D-%H:%M:%S")
        opt_out_123 = optimize_3d(z_n_dict["Z_0"], z_n_dict["Z_1"], z_n_dict["Z_2"],
                                  z_n_dict["N_0"], z_n_dict["N_1"], z_n_dict["N_2"],
                                  p_1, p_2, p_3, sb2_1, sb2_2, sb2_3, s02_1, s02_2, s02_3,
                                  p_12, p_13, p_23, rho_12, rho_13, rho_23,  rho0_12, rho0_13, rho0_23,
                                  r2_het_hist, ld_scores, nbin_het_hist, config["optimization"]["maxiter_3d"])
        now = datetime.now()
        end_time = now.strftime("%D-%H:%M:%S")
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Trivariate result 1 vs 2 vs 3:")
        print(opt_out_123)    

    out_dict = dict(config=config, opt_out_1=opt_out_1, opt_out_2=opt_out_2, opt_out_3=opt_out_3,
                    opt_out_12=opt_out_12, opt_out_13=opt_out_13, opt_out_23=opt_out_23, opt_out_12_13_23=opt_out_12_13_23,
                    opt_out_123=opt_out_123)

    with open(config["out"], 'w') as f:
        json.dump(out_dict, f, indent=4)

    print("Done!")
