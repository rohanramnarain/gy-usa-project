import re, pandas as pd

df = pd.read_csv("october_2025_stem.csv")

def pick(cols, *patterns):
    for pat in patterns:
        for c in cols:
            if re.search(pat, c, flags=re.I):
                return c
    return None

# Try to detect common column names across STEM/non-STEM months
country_col = pick(df.columns, r'\bCoC\b', r'Country of Citizenship', r'\bCountry\b')
state_col   = pick(df.columns, r'StateAbbrv', r'\bState\b', r'U\.?S\.?\s*State')
school_col  = pick(df.columns, r'\bU\.?S\.?\s*School\b', r'\bSchool Name\b', r'\bSchool\b')
count_col   = pick(df.columns, r'Count of Active STEM Students', r'Active Students', r'\bCount\b')

if not country_col or not count_col:
    raise SystemExit(f"Couldn’t find country/count columns. Got: {list(df.columns)}")

# Filter to Guyana (case-insensitive)
mask = df[country_col].astype(str).str.casefold().eq("guyana")
guy = df.loc[mask].copy()

# --- Always write state-level summary ---
if not state_col:
    # Some tables might not have state — fall back to total by country
    state_out = guy.groupby([], as_index=False)[count_col].sum()  # total only
    state_out.rename(columns={count_col: "total"}, inplace=True)
    state_out.to_csv("october_2025_stem_guyana_total.csv", index=False)
else:
    state_out = (guy.groupby(state_col, as_index=False)[count_col].sum()
                   .rename(columns={state_col: "state", count_col: "total"}))
    state_out.to_csv("october_2025_stem_guyana_by_state.csv", index=False)

print("Wrote state summary.")

# --- If a school column exists, also write school-level summary ---
if school_col:
    if not state_col:
        school_out = (guy.groupby(school_col, as_index=False)[count_col].sum()
                        .rename(columns={school_col:"institution", count_col:"total"}))
    else:
        school_out = (guy.groupby([school_col, state_col], as_index=False)[count_col].sum()
                        .rename(columns={school_col:"institution", state_col:"state", count_col:"total"}))
    school_out.to_csv("october_2025_stem_guyana_by_school.csv", index=False)
    print("Wrote school-level file.")
else:
    print("No school column on this snapshot—school-level file not created.")
