import pandas as pd
import numpy as np
import gzip
import os
from collections import defaultdict
import sys
import argparse

example_cmd =  """Example:
python make_template.py --bim ../data/chr@ --ld ../data/chr@.r2 --frq ../data/chr@.maf --chr 21 22"""

def parse_args(args):
    parser = argparse.ArgumentParser(description="A tool to create template for triplemix.", epilog=example_cmd)
    parser.add_argument("--bim", required=True, help="Prefix of plink bim files. '@' will be replace with chr ids")
    parser.add_argument("--ld", required=True, help="Prefix of plink ld files. '@' will be replace with chr ids")
    parser.add_argument("--frq", required=True, help="Prefix of plink frq files. '@' will be replace with chr ids")
    parser.add_argument("--chr", type=int, default=list(range(1,23)), choices=list(range(1,23)), nargs='+',
            help="Space separated list of chr ids to replace '@' in bim, ld and frq arguments. Default 1...22")
    parser.add_argument("--out", default='./', help="Output directory path. Default './'")
    return parser.parse_args(args)


def read_snps(chromosomes, bim_files, frq_files):
    # Args:
    #     chromosomes: list of chrom numbers (str)
    #     bim_files: dict with chr:bim_file
    #     frq_files: dict with chr:frq_file
    # Returns:
    #     snps: DataFrame with SNPs info: CHR, SNP, BP, A1, A2, MAF
    snps = pd.concat([pd.read_csv(bim_files[c], sep='\t', header=None, names=['CHR','SNP','BP','A1','A2'],
        usecols=[0,1,3,4,5], dtype={'CHR':'i4'}, na_filter=False, engine='c') for c in chromosomes], ignore_index=True)
    frq = pd.concat([pd.read_csv(frq_files[c], delim_whitespace=True, usecols=['SNP','MAF'], na_filter=False, engine='c')
        for c in chromosomes], ignore_index=True)
    snps = snps.merge(frq, on='SNP', how='inner', sort=False)
    assert snps.shape[0] == frq.shape[0]
    return snps


# https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
def process_ld(ld_files, chrom, snps, out_dir):
    # Args:
    #     ld_files: dict with chr:ld_file
    #     chrom: chromosome number
    #     snps: DataFrame produced with read_snps func.
    #     out_dir: directory where produced files are placed
    # Result:
    #     Produces 3 files for a given chrom:
    #         chr<chrom>.snps.gz - contains info for SNPs of the given chrom,
    #                              including LD_N - number of LD neighbours
    #         chr<chrom>.ld_r2 - contains r2 values
    #         chr<chrom>.ld_idx - contains indices of LD neighbours in corresponding bim file
    snps_chr = snps.loc[snps.CHR==chrom,:].copy()
    snp_idx_dict = dict(zip(snps_chr.SNP, range(snps_chr.shape[0])))
    snp_r2_dict, snp_snp_dict = defaultdict(list), defaultdict(list)
    r2_outf_name = os.path.join(out_dir, f'chr{chrom}.ld_r2')
    idx_outf_name = os.path.join(out_dir, f'chr{chrom}.ld_idx')
    snp_outf_name = os.path.join(out_dir, f'chr{chrom}.snp.gz')
    r2_outf = open(r2_outf_name, 'wb')
    idx_outf = open(idx_outf_name, 'wb')
    ld_n = []
    with gzip.open(ld_files[chrom], 'rt') as f:
        header = f.readline().split()
        ia, ib, ir2 = [header.index(name) for name in ['SNP_A', 'SNP_B', 'R2']]
        current_i_sa = 0
        for l in f:
            ll = l.split()
            i_sa, i_sb, r2 = snp_idx_dict[ll[ia]], snp_idx_dict[ll[ib]], ll[ir2]
            if i_sa > current_i_sa:
                process_current(current_i_sa, i_sa, snp_r2_dict, snp_snp_dict, ld_n, idx_outf, r2_outf)
                current_i_sa = i_sa
            snp_r2_dict[i_sa].append(r2)
            snp_r2_dict[i_sb].append(r2)
            snp_snp_dict[i_sa].append(i_sb)
            snp_snp_dict[i_sb].append(i_sa)
    i_sa = snps_chr.shape[0] 
    process_current(current_i_sa, i_sa, snp_r2_dict, snp_snp_dict, ld_n, idx_outf, r2_outf)
    assert len(snp_snp_dict) == 0
    snps_chr['LD_N'] = np.array(ld_n, dtype='i4')
    snps_chr.to_csv(snp_outf_name, index=False, sep='\t')
    r2_outf.close()
    idx_outf.close()

def process_current(current_i_sa, i_sa, snp_r2_dict, snp_snp_dict, ld_n, idx_outf, r2_outf):
    for i_s in range(current_i_sa, i_sa):
        snp_r2_dict[i_s].append(1)
        snp_snp_dict[i_s].append(i_s)
        snp_arr = np.array(snp_snp_dict[i_s], dtype='i4')
        r2_arr = np.array(snp_r2_dict[i_s], dtype='f4')
        i_sort = np.argsort(snp_arr)
        snp_arr = snp_arr[i_sort]
        ld_n.append(snp_arr.shape[0])
        r2_arr = r2_arr[i_sort]
        idx_outf.write(snp_arr.data)
        r2_outf.write(r2_arr.data)
        del snp_r2_dict[i_s]
        del snp_snp_dict[i_s]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    bim_files = {c:f"{args.bim.replace('@',str(c))}.bim" for c in args.chr} # '.bim' is added
    ld_files = {c:f"{args.ld.replace('@',str(c))}.ld.gz" for c in args.chr} # '.ld.gz' is added
    frq_files = {c:f"{args.frq.replace('@',str(c))}.frq" for c in args.chr} # '.frq' is added

    snps = read_snps(args.chr, bim_files, frq_files)
    out_dir = args.out
    for c in args.chr:
        print(f'Processing chr {c}')
        process_ld(ld_files, c, snps, out_dir)

    print("Done")
