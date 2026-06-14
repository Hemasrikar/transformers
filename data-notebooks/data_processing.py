# # Global Factor EM - Preprocessing Pipeline
# 
# **Dataset:** Jensen, Kelly and Pedersen (2023) Global Factor Data, Emerging Markets subset  
# **Panel:** ~4.4 M stock-month observations, 24 MSCI EM countries, 1995–2025
# 
# 
# - Train: 1995–2015 for Model training 
# - Validation: 2016–2020 for Hyperparameter selection 
# - Test: 2021–2025 for Out-of-sample evaluation 
# 
# **Processing steps**
# 1. Load data with memory optimisation  
# 2. Sort panel by `(id, eom)`  
# 3. Compute multi-horizon cumulative excess-return targets (3, 6, 12 months)  
# 4. Classify characteristics into **K0** (market-based) and **K1** (accounting-based)  
# 5. Column-level missing filter: computed on training rows only  
# 6. Build annual lag columns for retained K1 characteristics  
# 7. Split into train / val / test  
# 8. Row-level missing filter: drop firm-months with >1/3 original characteristics missing  
# 9. Add binary missingness flags for original characteristics  
# 10. Cross-sectional median imputation and rank normalisation  
# 11. Save processed splits and column metadata

import json
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
import gc

warnings.filterwarnings('ignore')

## Configuration

data_path = Path(r'data\Global Factor_IND.parquet')
output_dir = Path(r'data\processed')
output_dir.mkdir(parents=True, exist_ok=True)

TRAIN_END = '2015-12-31'
VAL_END = '2020-12-31'
MISSING_COL_THRESHOLD = 0.30          # drop columns with >30 % missing in training set
K1_LAG_MONTHS = [12, 24, 36, 48, 60]
FORWARD_HORIZONS = [3, 6, 12]    # forecast horizons in months
LAG_BATCH_SIZE = 20          # K1 columns processed per merge batch

## Column Definitions
# Panel identifiers, raw returns, and raw fundamentals that are excluded from the characteristic processing steps.

METADATA_COLS = [
    'permno', 'permco', 'gvkey', 'iid', 'id', 'date', 'excntry', 'eom',
    'obs_main', 'exch_main', 'common', 'primary_sec', 'source_crsp',
    'size_grp', 'me', 'me_company', 'prc', 'prc_local', 'prc_high',
    'prc_low', 'bidask', 'curcd', 'fx', 'gics', 'naics', 'sic', 'ff49',
    'dolvol', 'shares', 'tvol', 'adjfct', 'comp_tpci', 'crsp_shrcd',
    'comp_exchg', 'crsp_exchcd', 'ret', 'ret_exc', 'ret_local',
    'ret_exc_lead1m', 'ret_lag_dif', 'enterprise_value', 'book_equity',
    'assets', 'sales', 'net_income', 'intrinsic_value',
]

# K0: market-based, updated at monthly or daily frequency — no annual lags.
# K1 is the complement: all retained characteristics not listed here.
K0_CHARACTERISTICS = [
    'market_equity',

    # Dividend yields
    'div1m_me', 'div3m_me', 'div6m_me', 'div12m_me',
    'divspc1m_me', 'divspc12m_me',

    # Share changes
    'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'chcsho_12m',

    # Equity net payout (market-based)
    'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'eqnpo_12m',

    # Momentum and reversal (already embed historical return information)
    'ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1',
    'ret_12_7', 'ret_60_12', 'ret_2_0', 'ret_3_0', 'ret_6_0',
    'ret_9_0', 'ret_12_0', 'ret_18_1', 'ret_24_1', 'ret_24_12',
    'ret_36_1', 'ret_36_12', 'ret_48_1', 'ret_48_12',
    'ret_60_1', 'ret_60_36',

    # Seasonality (computed over multi-year windows by construction)
    'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',

    # Residual momentum
    'resff3_6_1', 'resff3_12_1',

    # Idiosyncratic volatility and skewness (daily/monthly data)
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_capm_60m',
    'ivol_ff3_21d', 'ivol_hxz4_21d',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',

    # Realised volatility
    'rvol_21d', 'rvol_252d', 'rvolhl_21d',

    # Maximum return
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d',

    # Skewness
    'rskew_21d', 'coskew_21d',

    # Beta measures
    'beta_60m', 'beta_21d', 'beta_252d',
    'beta_dimson_21d', 'betadown_252d', 'betabab_1260d',

    # Liquidity
    'ami_126d', 'dolvol_126d', 'dolvol_var_126d',
    'turnover_126d', 'turnover_var_126d',
    'zero_trades_21d', 'zero_trades_126d', 'zero_trades_252d',
    'bidaskhl_21d', 'corr_1260d',

    # Price to 52-week high
    'prc_highprc_252d',

    # Firm age (increments monthly)
    'age',

    # Market-asset-based liquidity
    'aliq_mat',

    # Mispricing composites (updated at monthly frequency)
    'mispricing_mgmt', 'mispricing_perf',
]

print(f"Metadata columns: {len(METADATA_COLS)}")
print(f"K0 (market-based): {len(K0_CHARACTERISTICS)}")

## Load Data
# Coerce string-encoded numeric columns (`divspc1m_me`, `divspc12m_me`) 
# and downcast all characteristic float64 columns to float32, reducing in-memory size by approximately 50 percent.

def load_data(path):
    print("Loading data")
    df = pd.read_parquet(path)

    # These two columns are stored as strings in the parquet schema
    for col in ['divspc1m_me', 'divspc12m_me']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep key return and price columns in float64 for precision
    keep_f64 = {
        'me', 'me_company', 'ret', 'ret_exc', 'ret_local',
        'ret_exc_lead1m', 'prc', 'prc_local', 'fx',
    }
    f32_cols = [c for c in df.select_dtypes('float64').columns if c not in keep_f64]
    df[f32_cols] = df[f32_cols].astype('float32')

    df['eom'] = pd.to_datetime(df['eom'])
    df['date'] = pd.to_datetime(df['date'])

    print(f" Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    return df


df = load_data(data_path)

## Sort Panel and Deduplicate
# 
# Sort by `(id, eom)` and remove any duplicate `(id, eom)` observations. 
# Duplicates can arise when annual and quarterly accounting update records overlap for the same security in the same month. 
# A unique panel is required for the indexed lag lookup in Step 6 and for the cross-sectional normalisation in Step 9 to behave correctly.


def sort_panel(df):
    df = df.sort_values(['id', 'eom']).reset_index(drop=True)

    # Remove duplicate (id, eom) observations that can arise from
    # overlapping annual and quarterly accounting update records.
    # The panel must be unique on (id, eom) for lag construction
    # and cross-sectional operations to behave correctly

    n_before = len(df)
    df = df.drop_duplicates(subset=['id', 'eom'], keep='first')
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f'Removed {n_dupes:,} duplicate (id, eom) observations')

    print(f'Shape after dedup: {df.shape[0]:,} rows x {df.shape[1]} columns')
    
    return df

df = sort_panel(df)

## Return Column Diagnostic
# 
# Before computing cumulative targets, we verify that `ret_exc` is in decimal
# form (0.05 = 5%) rather than percentage form (5.0 = 5%). If the median
# absolute value exceeds 0.5, the column is almost certainly in percentage
# form and is rescaled by dividing by 100. Monthly returns are then winsorised
# at the 0.1st and 99.9th percentiles to prevent a small number of extreme
# observations from dominating the compounded target values.

# Diagnostic: check ret_exc scale
ret = df['ret_exc'].dropna()
print(f"ret_exc before any adjustment")
print(f"count:{len(ret):,}")
print(f"mean: {ret.mean():.6f}")
print(f"std: {ret.std():.6f}")
print(f"min: {ret.min():.4f}")
print(f"1st: {ret.quantile(0.01):.4f}")
print(f"50th: {ret.quantile(0.50):.4f}")
print(f"99th: {ret.quantile(0.99):.4f}")
print(f"max: {ret.max():.4f}")

# Automatic rescaling: if the median absolute value exceeds 0.5,
# returns are in percentage form
if ret.abs().median() > 0.5:
	print("\nDetected percentage form. Rescaling ret_exc by 1/100.")
	df['ret_exc'] = df['ret_exc'] / 100
	ret = df['ret_exc'].dropna()
	print(f"mean after rescaling: {ret.mean():.6f}")
	print(f"std after rescaling:  {ret.std():.6f}")
else:
	print("\nReturns appear to be in decimal form. No rescaling needed.")

# Winsorise at 0.1/99.9 percentiles to cap extreme monthly returns
# before they enter the compounding formula
lo = df['ret_exc'].quantile(0.001)
hi = df['ret_exc'].quantile(0.999)
n_clipped = ((df['ret_exc'] < lo) | (df['ret_exc'] > hi)).sum()
df['ret_exc'] = df['ret_exc'].clip(lo, hi)
print(f"\nWinsorised ret_exc at [{lo:.4f}, {hi:.4f}]")
print(f"{n_clipped:,} observations clipped ({n_clipped / len(df):.2%})")

## Compute Multi-Horizon Return Targets

def compute_return_targets(df, horizons):
	df = df.copy()
	max_h = max(horizons)

	log_shifts = {
		s: np.log1p(df.groupby('id')['ret_exc'].shift(-s))
		for s in range(1, max_h + 1)
	}

	for h in horizons:
		col = f'target_{h}m'
		df[col] = np.expm1(
			sum(log_shifts[s] for s in range(1, h + 1))
		).astype('float32')

	return df


df = compute_return_targets(df, FORWARD_HORIZONS)

for h in FORWARD_HORIZONS:
	col = f'target_{h}m'
	data = df[col].dropna()
	print(f"{col}: {data.count():,} non-null ({data.count() / len(df):.1%}), "
	      f"mean={data.mean():.4f}, std={data.std():.4f}")

# Sanity check: if the mean absolute target exceeds 2.0 for the shortest
# horizon, something is wrong with the return scale
shortest = f'target_{min(FORWARD_HORIZONS)}m'
check_mean = df[shortest].dropna().abs().mean()
assert check_mean < 2.0, (
	f"Target sanity check failed: mean |{shortest}| = {check_mean:.2f}. "
	f"This suggests ret_exc is not in decimal form or contains extreme outliers."
)

## Classify Characteristics
# K0 is the explicitly listed set of market-based characteristics. 
# K1 is the complement: all remaining columns that are not metadata, return columns, or targets.

def classify_characteristics(df, metadata_cols, k0_list):
    exclude = set(metadata_cols) | {c for c in df.columns if c.startswith('target_')}
    all_chars = [c for c in df.columns if c not in exclude]
    k0_cols = [c for c in k0_list  if c in set(all_chars)]
    k1_cols = [c for c in all_chars if c not in set(k0_cols)]
    
    return k0_cols, k1_cols


k0_cols, k1_cols = classify_characteristics(df, METADATA_COLS, K0_CHARACTERISTICS)
print(f"K0 (market-based): {len(k0_cols)}")
print(f"K1 (accounting-based): {len(k1_cols)}")
print(f"Total characteristics: {len(k0_cols) + len(k1_cols)}")

## Column-Level Missing Filter
# Null rates are computed using training-set rows only (eom <= 2015-12-31).
# Characteristics exceeding the threshold are discarded. The same column list is applied to all splits, preventing any look-ahead in feature selection.
# Rejected columns are dropped from the full dataframe before lag construction, which reduces memory usage in Step 6.

def filter_columns_by_missing(df, train_end, k0_cols, k1_cols, threshold):
    train_mask = df['eom'] <= pd.Timestamp(train_end)
    null_rates = df.loc[train_mask, k0_cols + k1_cols].isnull().mean()

    retained_k0 = [c for c in k0_cols if null_rates[c] <= threshold]
    retained_k1 = [c for c in k1_cols if null_rates[c] <= threshold]

    print(f"K0: {len(retained_k0)} retained ({len(k0_cols) - len(retained_k0)} dropped)")
    print(f"K1: {len(retained_k1)} retained ({len(k1_cols) - len(retained_k1)} dropped)")
    return retained_k0, retained_k1


retained_k0, retained_k1 = filter_columns_by_missing(
    df, TRAIN_END, k0_cols, k1_cols, MISSING_COL_THRESHOLD
)

# Drop rejected columns immediately to reduce memory before lag construction
rejected = (set(k0_cols) - set(retained_k0)) | (set(k1_cols) - set(retained_k1))
df = df.drop(columns=list(rejected), errors='ignore')
print(f"Dropped {len(rejected)} columns. Current shape: {df.shape[1]} columns")

## Build K1 Annual Lag Columns
# For each retained K1 characteristic, five annual lag columns are created at months 12, 24, 36, 48, 60.


def build_k1_lags(df, k1_cols, lag_months, batch_size=50):

    # The previous merge-based approach reindexed the FULL dataframe
    # (all columns, including datetimes) on every iteration, causing a
    # MemoryError once enough lag columns had accumulated.
    #
    # This version uses set_index + reindex (O(n) index lookup) instead
    # of merge, and assigns columns in-place to avoid full-df copies.
    #
    # The lookup is deduplicated on (id, _period) before indexing.
    # Duplicate (id, _period) pairs cause a non-unique MultiIndex
    # that reindex cannot handle; deduplication here is a defensive
    # guard even after sort_panel has already cleaned the main df

    df['_period'] = df['eom'].dt.year * 12 + df['eom'].dt.month

    # Deduplicate before indexing to guarantee a unique MultiIndex
    lookup = (df[['id', '_period'] + k1_cols].drop_duplicates(subset=['id', '_period']).set_index(['id', '_period']))

    n_batches = (len(k1_cols) + batch_size - 1) // batch_size

    for lag in lag_months:
        keys = pd.MultiIndex.from_arrays([df['id'].values, df['_period'].values - lag])

        for b, start in enumerate(range(0, len(k1_cols), batch_size)):
            batch = k1_cols[start : start + batch_size]
            lag_vals = lookup[batch].reindex(keys).values.astype('float32')

            for i, col in enumerate(batch):
                df[f'{col}_lag{lag}'] = lag_vals[:, i]

            del lag_vals

        print(f'Lag {lag}m done ({df.shape[1]} columns total)')

    del lookup
    
    return df.drop(columns=['_period'])


df = build_k1_lags(df, retained_k1, K1_LAG_MONTHS, LAG_BATCH_SIZE)

lag_cols = [f'{c}_lag{l}' for c in retained_k1 for l in K1_LAG_MONTHS]
orig_char_cols = retained_k0 + retained_k1
all_char_cols = orig_char_cols + lag_cols

print(f'Total characteristic columns: {len(all_char_cols)}')
print(f'K0 current: {len(retained_k0)}')
print(f'K1 current: {len(retained_k1)}')
print(f'K1 lag cols: {len(lag_cols)}')

## Date Split
# Split the full panel into three chronologically non-overlapping subsets. 
# The full-panel dataframe is deleted afterwards to free memory.

def split_data(df, train_end, val_end):
    t1 = pd.Timestamp(train_end)
    t2 = pd.Timestamp(val_end)

    train = df[df['eom'] <= t1].copy()
    val = df[(df['eom'] > t1) & (df['eom'] <= t2)].copy()
    test = df[df['eom'] > t2].copy()

    for name, split in [('Train', train), ('Val', val), ('Test', test)]:
        print(f'{name}: {split.shape[0]:,} rows ' f'({split["eom"].min().date()} to {split["eom"].max().date()})')

    return train, val, test


train, val, test = split_data(df, TRAIN_END, VAL_END)

# Free the full-panel dataframe and force garbage collection.
# The lag-construction step leaves many fragmented blocks. Releasing
# the reference and calling gc.collect() recovers that memory before
# the normalisation step runs

del df
gc.collect()

## Row-Level Missing Filter

def drop_high_missing_rows(df, char_cols, threshold = 1/3, label = ""):
	miss_frac = df[char_cols].isnull().mean(axis = 1)
	keep = miss_frac <= threshold
	n_drop = (~keep).sum()
	print(
		f"{label}: dropped {n_drop:,} rows ({n_drop / len(df):.2%}) "
		f"with >{threshold:.0%} missing characteristics"
	)
	return df.loc[keep].reset_index(drop = True)


train = drop_high_missing_rows(train, orig_char_cols, threshold = 1/3, label = "Train")
val = drop_high_missing_rows(val, orig_char_cols, threshold = 1/3, label = "Val")
test = drop_high_missing_rows(test, orig_char_cols, threshold = 1/3, label = "Test")

print(f"Train:{train.shape[0]:,} rows")
print(f"Val:{val.shape[0]:,} rows")
print(f"Test:{test.shape[0]:,} rows")

## Binary Missingness Flags
# For each original (non-lag) characteristic, a binary flag column (suffix `_miss`) is added. 
# Flags are constructed **before** any imputation so that the model receives the genuine missingness 
# signal as a separate input alongside the characteristic value. 
# This allows the attention mechanism to learn whether the absence of a characteristic carries predictive content.

def add_missingness_flags(df, orig_char_cols):
    flags = (df[orig_char_cols].isnull().astype('float32').rename(columns={c: f'{c}_miss' for c in orig_char_cols}))

    return pd.concat([df, flags], axis=1)

train = add_missingness_flags(train, orig_char_cols)
val = add_missingness_flags(val, orig_char_cols)
test = add_missingness_flags(test, orig_char_cols)

flag_cols = [c for c in train.columns if c.endswith('_miss')]  # type: ignore
print(f"Missingness flag columns added: {len(flag_cols)}")
print(f"Train shape after flags: {train.shape}")

## Cross-Sectional Normalisation

def cross_sectional_normalise(df, char_cols, verbose_every=50):
    # Extract char_cols as a single numpy array once
    # All per-month operations run on this array in-place

    data = df[char_cols].to_numpy(dtype='float32', na_value=np.nan)
    eoms = df['eom'].to_numpy()
    unique_eoms = np.unique(eoms)
    n_months = len(unique_eoms)

    for i, eom in enumerate(unique_eoms):
        mask = eoms == eom
        xs = data[mask]
        N = xs.shape[0]

        if N == 0:
            continue

        # Vectorised median imputation
        col_medians = np.nanmedian(xs, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(xs))
        xs[nan_rows, nan_cols] = col_medians[nan_cols]

        # Rank normalisation with average-rank tie handling
        if N > 1:
            ranks = rankdata(xs, method='average', axis=0) - 1
            xs = (ranks / (N - 1) - 0.5).astype('float32')
        else:
            xs = np.zeros_like(xs)

        data[mask] = xs

        if (i + 1) % verbose_every == 0 or (i + 1) == n_months:
            print(f'{i + 1}/{n_months} months done')

    # Assign back in-place — avoids df.copy() which triggers block
    # consolidation and tries to allocate a single contiguous array
    # for all columns (~7 GB), causing the MemoryError.
    df[char_cols] = data
    del data
    gc.collect()
    
    return df


print('Normalising training set')
train = cross_sectional_normalise(train, all_char_cols)
gc.collect()

print('Normalising validation set')
val   = cross_sectional_normalise(val, all_char_cols)
gc.collect()

print('Normalising test set')
test  = cross_sectional_normalise(test, all_char_cols)
gc.collect()

## Save

def save_split(df, name, output_dir):
	path = output_dir / f'{name}.parquet'
	df.to_parquet(path, index = False)
	size_mb = path.stat().st_size / 1e6
	print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols  ({size_mb:.0f} MB)")


print('Saving splits')
save_split(train, 'train', output_dir)
save_split(val, 'val', output_dir)
save_split(test, 'test', output_dir)

# Build country lookup from the processed splits.
# excntry is a metadata column retained throughout the pipeline.
# Saving a dedicated lookup file removes the dependency on the raw
# parquet in all downstream model and tuning scripts.
all_excntry = pd.concat([
	train[['id', 'eom', 'excntry']],
	val[['id', 'eom', 'excntry']],
	test[['id', 'eom', 'excntry']],
], ignore_index = True).drop_duplicates()

COUNTRY_CODES = sorted(all_excntry['excntry'].dropna().unique().tolist())
COUNTRY_TO_ID = {c: i for i, c in enumerate(COUNTRY_CODES)}
all_excntry['country_id'] = all_excntry['excntry'].map(COUNTRY_TO_ID).astype('Int16')

country_lookup = all_excntry[['id', 'eom', 'country_id']].dropna(subset = ['country_id'])
country_lookup_path = output_dir / 'country_lookup.parquet'
country_lookup.to_parquet(country_lookup_path, index = False)
print(f"Country lookup saved: {country_lookup_path}  ({len(country_lookup):,} rows, {len(COUNTRY_CODES)} countries)")
del all_excntry, country_lookup
gc.collect()

# Column metadata and country mapping
col_metadata = {
	'retained_k0': retained_k0,
	'retained_k1': retained_k1,
	'lag_cols': lag_cols,
	'orig_char_cols': orig_char_cols,
	'all_char_cols': all_char_cols,
	'country_to_id': COUNTRY_TO_ID,
	'country_codes': COUNTRY_CODES,
}

meta_path = output_dir / 'column_metadata.json'
with open(meta_path, 'w') as f:
	json.dump(col_metadata, f, indent = 2)

print(f'Column metadata saved: {meta_path}')
print(f'Countries ({len(COUNTRY_CODES)}):', ', '.join(COUNTRY_CODES))


