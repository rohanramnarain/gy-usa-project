import glob
import pandas as pd

files = sorted(glob.glob("*.csv"))  # change the pattern if needed
dfs = []
for f in files:
    df = pd.read_csv(f)            # add arguments like encoding="utf-8" if required
    dfs.append(df)

out = pd.concat(dfs, ignore_index=True)  # outer-joins columns by default
out.to_csv("guyana_state_by_month.csv", index=False)
print("Wrote guyana_state_by_month.csv with", len(out), "rows")