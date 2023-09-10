import sys
import pandas as pd
import json
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description="Combine parameters from multiple mix3r runs.")
    parser.add_argument("--input", required=True, nargs='+', help="A list of mix3r output json files.")
    parser.add_argument("--out", required=True, help="Output file prefix.")
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])
fnames = args.input
print(f"Loading {fnames}")
outf = f"{args.out}.parameters.csv" # "aud_adhd_mig_aug22.parameters.csv"

df = pd.DataFrame(columns="run_id p_1 sb2_1 s02_1 success_1 p_2 sb2_2 s02_2 success_2 p_3 sb2_3 s02_3 success_3 p_12 rho_12 rho0_12 success_12 p_13 rho_13 rho0_13 success_13 p_23 rho_23 rho0_23 success_23 p_123 success_123".split())

for irow, fname in enumerate(fnames):
    row = [fname]
    with open(fname) as f:
        d = json.load(f)
        for i in "1 2 3".split():
            k = f"opt_out_{i}"
            row += d[k]["opt_par"]
            row.append(d[k]["opt_res"]["success"])
        if not d["opt_out_12_13_23"] is None:
            p12, p13, p23, rho12, rho13, rho23, rho012, rho013, rho023 = d["opt_out_12_13_23"]["opt_par"]
            success = d["opt_out_12_13_23"]["opt_res"]["success"]
            row += [p12, rho12, rho012, success, p13, rho13, rho013, success, p23, rho23, rho023, success]
        else:
            for i in "12 13 23".split():
                k = f"opt_out_{i}"
                row += d[k]["opt_par"]
                row.append(d[k]["opt_res"]["success"])
        i = "123"
        k = f"opt_out_{i}"
        row += d[k]["opt_par"]
        row.append(d[k]["opt_res"]["success"])
        df.loc[irow] = row

pcols = "p_1 p_2 p_3 p_12 p_13 p_23 p_123".split()
total = df["p_1"] + df["p_2"] + df["p_3"] - df["p_12"] - df["p_13"] - df["p_23"] + df["p_123"]
df_p_proportion = df[pcols].div(total,axis=0)
i_min = (df_p_proportion - df_p_proportion.median()).abs().sum(axis=1).argmin()
i_rank = (df_p_proportion - df_p_proportion.median()).abs().sum(axis=1).rank()
df["rank_p_proportion_deviation_from_median"] = i_rank
print(df)
print(f"Minimum deviation from median p proportions in run {i_min}")
df.to_csv(outf, sep='\t', index=False)
print(f"{outf} saved.")
