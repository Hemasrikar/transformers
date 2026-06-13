## Data Processing: Single Country JKP Dataset

# Processes one country's raw JKP parquet: coverage filter, per stock missing filter,
# within country rank normalisation to [-0.5, 0.5], median imputation. 
# JKP sample screens were applied at download time from WRDS and do not require reapplication.

## Setup

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# Set these two variables for your country
country = 'CHN'
raw_path = Path('../data/Global Factor_CHN.parquet')

output_dir = Path('../data/processed') / country
output_dir.mkdir(parents = True, exist_ok = True)

coverage_threshold = 0.70
max_miss_frac = 1.0 / 3.0
min_stocks = 30

train_end = '2014-12-31'
val_end = '2019-12-31'

load_always = ['id', 'gvkey', 'eom', 'excntry', 'ret_exc_lead1m', 'me']

exclude_cols = {
	'id', 'gvkey', 'iid', 'permno', 'permco', 'date', 'eom', 'excntry',
	'size_grp', 'obs_main', 'exch_main', 'common', 'primary_sec',
	'source_crsp', 'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
	'curcd', 'fx', 'adjfct', 'bidask',
	'ret', 'ret_local', 'ret_exc', 'ret_exc_lead1m',
	'prc', 'prc_local', 'prc_high', 'prc_low',
	'me', 'me_company', 'dolvol', 'shares', 'tvol',
	'ret_lag_dif', 'div_tot',
}

print(f'country: {country}')


### Load Raw Data

schema = pq.read_schema(raw_path)
all_col_names = schema.names
print(f'total columns: {len(all_col_names)}')

char_candidate = [
	c for c in all_col_names
	if c not in exclude_cols and c not in load_always
]
print(f'characteristic candidates: {len(char_candidate)}')

needed = [c for c in load_always + char_candidate if c in all_col_names]
df = pd.read_parquet(raw_path, columns = needed)
df['eom'] = pd.to_datetime(df['eom'])
print(f'loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'date range: {df["eom"].min().date()} to {df["eom"].max().date()}')

for col in char_candidate:
	if col in df.columns and df[col].dtype == np.float64:
		df[col] = df[col].astype(np.float32)
if 'me' in df.columns and df['me'].dtype == np.float64:
	df['me'] = df['me'].astype(np.float32)

char_candidate = [c for c in char_candidate if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
print(f'numeric candidates: {len(char_candidate)}')
gc.collect()


## Coverage Filter

df_tr = df[df['eom'] <= train_end]
coverage = df_tr[char_candidate].notna().mean()
char_cols = sorted([c for c in char_candidate if coverage[c] >= coverage_threshold])
d = len(char_cols)
print(f'features with >= {coverage_threshold:.0%} coverage: d = {d}')

id_cols = [c for c in load_always if c in df.columns]
df = df[id_cols + char_cols]
del df_tr
gc.collect()


### Missing Filter, Rank Normalisation, Save

# Missing filter
n_miss = df[char_cols].isna().sum(axis = 1)
df = df[n_miss <= d * max_miss_frac].reset_index(drop = True)
print(f'after missing filter: {len(df):,} rows')

# Firm lookup (reset_index before column selection to keep id)
info_cols = [c for c in ['id', 'gvkey', 'excntry', 'me'] if c in df.columns]
if 'me' in df.columns and df['me'].notna().any():
	select_cols = [c for c in info_cols if c != 'id']
	firm_lookup = (
		df[df['me'].notna()]
		.sort_values('eom')
		.groupby('id')
		.last()[select_cols]
		.reset_index()
	)
else:
	firm_lookup = df[info_cols].drop_duplicates(subset = ['id'])
if 'excntry' not in firm_lookup.columns:
	firm_lookup['excntry'] = country

# Per month rank normalisation
processed = []
for eom in sorted(df['eom'].unique()):
	month = df[df['eom'] == eom].copy()
	if len(month) < min_stocks: continue
	ranks = month[char_cols].rank(pct = True, axis = 0) - 0.5
	month[char_cols] = ranks.fillna(0.0)
	processed.append(month)
df_proc = pd.concat(processed, ignore_index = True)
print(f'processed: {len(df_proc):,} rows, {df_proc["eom"].nunique()} months')
print(f'mean firms per month: {len(df_proc) / df_proc["eom"].nunique():.0f}')

# Split
tr = df_proc[df_proc['eom'] <= train_end]
vl = df_proc[(df_proc['eom'] > train_end) & (df_proc['eom'] <= val_end)]
te = df_proc[df_proc['eom'] > val_end]

# Save
tr.to_parquet(output_dir / f'{country}_train.parquet', index = False)
vl.to_parquet(output_dir / f'{country}_val.parquet', index = False)
te.to_parquet(output_dir / f'{country}_test.parquet', index = False)
firm_lookup.to_parquet(output_dir / f'{country}_firm_lookup.parquet', index = False)

metadata = {
	'char_cols': char_cols, 'd': d, 'country': country,
	'coverage_threshold': coverage_threshold,
	'train_end': train_end, 'val_end': val_end,
}
with open(output_dir / f'{country}_metadata.json', 'w') as f:
	json.dump(metadata, f, indent = 2)

print(f'train: {len(tr):,} ({tr["eom"].min().date()} to {tr["eom"].max().date()})')
print(f'val:{len(vl):,} ({vl["eom"].min().date()} to {vl["eom"].max().date()})')
print(f'test:{len(te):,} ({te["eom"].min().date()} to {te["eom"].max().date()})')
print(f'firms:{len(firm_lookup):,}')
print(f'saved to {output_dir}')


### Sanity Checks

sample_eom = tr['eom'].iloc[100] if len(tr) > 100 else tr['eom'].iloc[0]
sample = tr[tr['eom'] == sample_eom][char_cols]
print(f'sample month {sample_eom.date()}: {len(sample)} firms')
print(f'feature range: [{sample.min().min():.4f}, {sample.max().max():.4f}]')
print(f'feature mean:{sample.mean().mean():.6f}')
print(f'nan count:{sample.isna().sum().sum()}')
print(f'ret missing:{tr["ret_exc_lead1m"].isna().mean():.2%}')


