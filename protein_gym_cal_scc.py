import pandas as pd
from scipy.stats import spearmanr
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True)
    args = parser.parse_args()

    score_csv_files = sorted(glob.glob(os.path.join(args.score_dir, "*.csv")))

    scc_ls = []
    dms_fname = []
    for file in score_csv_files:
        df = pd.read_csv(file)
        rho, pval = spearmanr(df["ALLY_score"], df["DMS_score"])
        scc_ls.append(rho)
        dms_fname.append(file.split('/')[-1])

    scc_df = pd.DataFrame({"File": dms_fname, "SCC": scc_ls})
    print(f"Aggregated SCC: {round(scc_df['SCC'].mean(), 4)}")

if __name__ == "__main__":
    main()