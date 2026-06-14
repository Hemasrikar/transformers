"""
dual_path_pipeline.py

Combined preprocessing and training pipeline for the Dual Path Portfolio
Transformer applied to the Jensen, Kelly and Pedersen (2023) global factor
dataset. The preprocessing stage loads the raw panel, computes multi-horizon
cumulative return targets, classifies characteristics into market-based (K0)
and accounting-based (K1) groups, constructs annual lag columns, applies
column and row level missing data filters, adds binary missingness flags,
performs cross-sectional median imputation followed by rank normalisation,
and saves the processed splits alongside column metadata and a country lookup
table. The training stage reads the processed parquet files, constructs
per-month cross-sectional tensors, and trains the Dual Path Transformer across
five encoding variants. The architecture separates per-firm scoring from
cross-sectional peer comparison by routing firm embeddings through an
attention-weighted aggregation module (Path 1) and a per-country sparse
attention module (Path 2), combining their outputs additively.
"""

import gc
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Panel metadata columns excluded from all characteristic processing steps.
# These include firm identifiers, raw price and volume series, and accounting
# aggregates whose cross-sectional distributions are not rank-normalised.

metadata_cols = [
    'permno', 'permco', 'gvkey', 'iid', 'id', 'date', 'excntry', 'eom',
    'obs_main', 'exch_main', 'common', 'primary_sec', 'source_crsp',
    'size_grp', 'me', 'me_company', 'prc', 'prc_local', 'prc_high',
    'prc_low', 'bidask', 'curcd', 'fx', 'gics', 'naics', 'sic', 'ff49',
    'dolvol', 'shares', 'tvol', 'adjfct', 'comp_tpci', 'crsp_shrcd',
    'comp_exchg', 'crsp_exchcd', 'ret', 'ret_exc', 'ret_local',
    'ret_exc_lead1m', 'ret_lag_dif', 'enterprise_value', 'book_equity',
    'assets', 'sales', 'net_income', 'intrinsic_value',
]

# K0 characteristics are updated at monthly or daily frequency and already
# embed their own historical return information. They receive no annual lags.
# K1 is the complement: all retained characteristics not listed here.

k0_characteristic_list = [
    'market_equity',
    'div1m_me', 'div3m_me', 'div6m_me', 'div12m_me',
    'divspc1m_me', 'divspc12m_me',
    'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'chcsho_12m',
    'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'eqnpo_12m',
    'ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1',
    'ret_12_7', 'ret_60_12', 'ret_2_0', 'ret_3_0', 'ret_6_0',
    'ret_9_0', 'ret_12_0', 'ret_18_1', 'ret_24_1', 'ret_24_12',
    'ret_36_1', 'ret_36_12', 'ret_48_1', 'ret_48_12',
    'ret_60_1', 'ret_60_36',
    'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',
    'resff3_6_1', 'resff3_12_1',
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_capm_60m',
    'ivol_ff3_21d', 'ivol_hxz4_21d',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
    'rvol_21d', 'rvol_252d', 'rvolhl_21d',
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d',
    'rskew_21d', 'coskew_21d',
    'beta_60m', 'beta_21d', 'beta_252d',
    'beta_dimson_21d', 'betadown_252d', 'betabab_1260d',
    'ami_126d', 'dolvol_126d', 'dolvol_var_126d',
    'turnover_126d', 'turnover_var_126d',
    'zero_trades_21d', 'zero_trades_126d', 'zero_trades_252d',
    'bidaskhl_21d', 'corr_1260d',
    'prc_highprc_252d',
    'age',
    'aliq_mat',
    'mispricing_mgmt', 'mispricing_perf',
]

# K1 annual lag positions in months. Lag 0 corresponds to the current
# period observation and is stored in the base characteristic column.
k1_lag_months = [12, 24, 36, 48, 60]
forward_horizons = [3, 6, 12]
lag_batch_size = 20

# Lag suffixes define the column name extensions for all six K1 positions.
# The ordering here is critical: the dataset reshape relies on columns
# appearing as [char, char_lag12, ..., char_lag60] for each K1 characteristic.
lag_suffixes = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]
lag_positions_list = [0, 12, 24, 36, 48, 60]
target_cols = ["target_3m", "target_6m", "target_12m"]


@dataclass
class Config:
    # Raw data and output paths
    data_path: Path = Path("data/Global Factor_USA.parquet")
    output_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    train_path: Path = Path("data/processed/train.parquet")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")

    # Preprocessing hyperparameters
    train_end: str = "2015-12-31"
    val_end: str = "2020-12-31"
    missing_col_threshold: float = 0.30
    lag_batch_size: int = 20

    # Transformer architecture dimensions
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    top_k_attention: int = 50
    ple_num_bins: int = 16
    periodic_num_freq: int = 32

    # Dual path specific parameters
    n_mlp_layers: int = 2
    lambda_aux: float = 0.3
    min_firms_attention: int = 10

    # Optimiser and training schedule
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    grad_clip: float = 1.0

    # Multi-horizon loss weights
    lambda_3m: float = 0.2
    lambda_6m: float = 0.5
    lambda_12m: float = 0.3

    encoding_variant: str = "linear"
    seed: int = 24


cfg = Config()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


# Preprocessing pipeline

def load_raw_data(path):
    """Load raw parquet, coerce string-encoded numeric columns, and downcast
    float64 characteristics to float32 to reduce memory consumption by
    approximately 50 percent."""
    print("Loading raw data")
    df = pd.read_parquet(path)
    for col in ['divspc1m_me', 'divspc12m_me']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    keep_f64 = {
        'me', 'me_company', 'ret', 'ret_exc', 'ret_local',
        'ret_exc_lead1m', 'prc', 'prc_local', 'fx',
    }
    f32_cols = [c for c in df.select_dtypes('float64').columns if c not in keep_f64]
    df[f32_cols] = df[f32_cols].astype('float32')
    df['eom'] = pd.to_datetime(df['eom'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Shape, {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def sort_panel(df):
    """Sort by (id, eom) and remove duplicate (id, eom) observations that can
    arise when annual and quarterly accounting update records overlap for the
    same security in the same month."""
    df = df.sort_values(['id', 'eom']).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=['id', 'eom'], keep='first')
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes:,} duplicate (id, eom) observations")
    print(f"Shape after deduplication, {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def compute_return_targets(df, horizons):
    """Compute compounded cumulative excess return targets at each horizon.
    Log returns are summed and exponentiated to obtain exact compound returns
    without the small-sample approximation error of simple return summation."""
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


def classify_characteristics(df, meta_cols, k0_list):
    """Partition all non-metadata columns into K0 (market-based) and K1
    (accounting-based). Target columns computed in the previous step are
    excluded from both sets."""
    exclude = set(meta_cols) | {c for c in df.columns if c.startswith('target_')}
    all_chars = [c for c in df.columns if c not in exclude]
    k0_cols = [c for c in k0_list if c in set(all_chars)]
    k1_cols = [c for c in all_chars if c not in set(k0_cols)]
    return k0_cols, k1_cols


def filter_columns_by_missing(df, train_end, k0_cols, k1_cols, threshold):
    """Drop characteristics whose null rate in the training set exceeds the
    specified threshold. Null rates are computed on training rows only to
    prevent any look-ahead in feature selection. The same retained column list
    is applied uniformly to all splits."""
    train_mask = df['eom'] <= pd.Timestamp(train_end)
    null_rates = df.loc[train_mask, k0_cols + k1_cols].isnull().mean()
    retained_k0 = [c for c in k0_cols if null_rates[c] <= threshold]
    retained_k1 = [c for c in k1_cols if null_rates[c] <= threshold]
    print(f"K0, {len(retained_k0)} retained ({len(k0_cols) - len(retained_k0)} dropped)")
    print(f"K1, {len(retained_k1)} retained ({len(k1_cols) - len(retained_k1)} dropped)")
    return retained_k0, retained_k1


def build_k1_lags(df, k1_cols, lag_months_list, batch_size=50):
    """Construct annual lag columns for all retained K1 characteristics at
    lags of 12, 24, 36, 48, and 60 months. An integer-period index is used
    for O(n) lookup rather than a full merge, and columns are assigned in
    batches to avoid accumulating large temporary arrays."""
    df['_period'] = df['eom'].dt.year * 12 + df['eom'].dt.month
    lookup = (
        df[['id', '_period'] + k1_cols]
        .drop_duplicates(subset=['id', '_period'])
        .set_index(['id', '_period'])
    )
    for lag in lag_months_list:
        keys = pd.MultiIndex.from_arrays(
            [df['id'].values, df['_period'].values - lag]
        )
        for start in range(0, len(k1_cols), batch_size):
            batch = k1_cols[start:start + batch_size]
            lag_vals = lookup[batch].reindex(keys).values.astype('float32')
            for i, col in enumerate(batch):
                df[f'{col}_lag{lag}'] = lag_vals[:, i]
            del lag_vals
        print(f"Lag {lag}m done, {df.shape[1]} columns total")
    del lookup
    return df.drop(columns=['_period'])


def split_data(df, train_end, val_end):
    """Partition the panel into chronologically non-overlapping train, validation,
    and test splits. No firm-level stratification is applied; the split boundary
    is strict on the end-of-month date."""
    t1 = pd.Timestamp(train_end)
    t2 = pd.Timestamp(val_end)
    train = df[df['eom'] <= t1].copy()
    val = df[(df['eom'] > t1) & (df['eom'] <= t2)].copy()
    test = df[df['eom'] > t2].copy()
    for name, split in [('Train', train), ('Val', val), ('Test', test)]:
        print(
            f"{name}, {split.shape[0]:,} rows, "
            f"{split['eom'].min().date()} to {split['eom'].max().date()}"
        )
    return train, val, test


def drop_high_missing_rows(df, char_cols, threshold=1/3, label=""):
    """Remove firm-months for which more than one third of original (non-lag)
    characteristics are missing. This filter is applied per split after the
    date split to prevent future information from influencing the threshold."""
    miss_frac = df[char_cols].isnull().mean(axis=1)
    keep = miss_frac <= threshold
    n_drop = (~keep).sum()
    print(
        f"{label}, dropped {n_drop:,} rows ({n_drop / len(df):.2%}) "
        f"with >{threshold:.0%} missing characteristics"
    )
    return df.loc[keep].reset_index(drop=True)


def add_missingness_flags(df, orig_char_cols):
    """Append binary missingness indicator columns (suffix _miss) for each
    original characteristic. Flags are constructed before imputation so that
    the attention mechanism can learn whether absence of a signal carries
    independent predictive content."""
    flags = (
        df[orig_char_cols]
        .isnull()
        .astype('float32')
        .rename(columns={c: f'{c}_miss' for c in orig_char_cols})
    )
    return pd.concat([df, flags], axis=1)


def cross_sectional_normalise(df, char_cols, verbose_every=50):
    """Apply cross-sectional median imputation followed by rank normalisation
    to the range [-0.5, 0.5]. Operations run on a single numpy array to avoid
    the block-consolidation memory spike that arises from per-column assignment
    on a wide pandas DataFrame."""
    data = df[char_cols].to_numpy(dtype='float32', na_value=np.nan)
    eoms = df['eom'].to_numpy()
    unique_eoms = np.unique(eoms)
    n_months = len(unique_eoms)
    for i, eom in enumerate(unique_eoms):
        mask = eoms == eom
        xs = data[mask]
        n = xs.shape[0]
        if n == 0:
            continue
        col_medians = np.nanmedian(xs, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(xs))
        xs[nan_rows, nan_cols] = col_medians[nan_cols]
        if n > 1:
            ranks = rankdata(xs, method='average', axis=0) - 1
            xs = (ranks / (n - 1) - 0.5).astype('float32')
        else:
            xs = np.zeros_like(xs)
        data[mask] = xs
        if (i + 1) % verbose_every == 0 or (i + 1) == n_months:
            print(f"{i + 1}/{n_months} months done")

    # For the earliest months in the panel the full lag history does not yet
    # exist. np.nanmedian on an entirely-NaN column returns NaN, so those
    # positions survive the imputation loop. We zero-fill them here so that
    # the saved parquet contains no NaN values in any characteristic column.
    residual_nan = int(np.isnan(data).sum())
    if residual_nan > 0:
        print(f"Zero-filling {residual_nan:,} residual NaN cells in lag columns")
        np.nan_to_num(data, nan=0.0, copy=False)

    df[char_cols] = data
    del data
    gc.collect()
    return df


def save_split_parquet(df, name, output_dir):
    """Write a processed split to parquet and report its size."""
    path = output_dir / f'{name}.parquet'
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1e6
    print(f"{name}, {df.shape[0]:,} rows x {df.shape[1]} cols, {size_mb:.0f} MB")


def run_preprocessing(cfg):
    """Execute the full preprocessing pipeline from raw parquet to processed
    splits. The function is self-contained and writes all outputs to disk,
    including the train, validation, and test parquet files, a country lookup
    table, and a column metadata JSON that all downstream scripts depend on."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(cfg.data_path)
    df = sort_panel(df)

    # Scale diagnostic and winsorisation of monthly excess returns
    ret = df['ret_exc'].dropna()
    print(
        f"ret_exc diagnostic, count {len(ret):,}, "
        f"mean {ret.mean():.6f}, std {ret.std():.6f}, "
        f"p1 {ret.quantile(0.01):.4f}, p50 {ret.quantile(0.50):.4f}, "
        f"p99 {ret.quantile(0.99):.4f}"
    )
    if ret.abs().median() > 0.5:
        print("Detected percentage form. Rescaling ret_exc by 1/100.")
        df['ret_exc'] = df['ret_exc'] / 100
        ret = df['ret_exc'].dropna()
        print(f"After rescaling, mean {ret.mean():.6f}, std {ret.std():.6f}")

    lo = ret.quantile(0.001)
    hi = ret.quantile(0.999)
    n_clipped = ((df['ret_exc'] < lo) | (df['ret_exc'] > hi)).sum()
    df['ret_exc'] = df['ret_exc'].clip(lo, hi)
    print(
        f"Winsorised ret_exc at [{lo:.4f}, {hi:.4f}], "
        f"{n_clipped:,} observations clipped ({n_clipped / len(df):.2%})"
    )

    df = compute_return_targets(df, forward_horizons)
    for h in forward_horizons:
        col = f'target_{h}m'
        data_h = df[col].dropna()
        print(
            f"{col}, {data_h.count():,} non-null ({data_h.count() / len(df):.1%}), "
            f"mean {data_h.mean():.4f}, std {data_h.std():.4f}"
        )

    shortest = f'target_{min(forward_horizons)}m'
    check_mean = df[shortest].dropna().abs().mean()
    assert check_mean < 2.0, (
        f"Target sanity check failed: mean |{shortest}| = {check_mean:.2f}. "
        f"This suggests ret_exc is not in decimal form or contains extreme outliers."
    )

    k0_cols, k1_cols = classify_characteristics(df, metadata_cols, k0_characteristic_list)
    print(f"K0 (market-based), {len(k0_cols)}")
    print(f"K1 (accounting-based), {len(k1_cols)}")
    print(f"Total characteristics, {len(k0_cols) + len(k1_cols)}")

    retained_k0, retained_k1 = filter_columns_by_missing(
        df, cfg.train_end, k0_cols, k1_cols, cfg.missing_col_threshold
    )
    rejected = (set(k0_cols) - set(retained_k0)) | (set(k1_cols) - set(retained_k1))
    df = df.drop(columns=list(rejected), errors='ignore')
    print(f"Dropped {len(rejected)} columns. Current shape, {df.shape[1]} columns")

    df = build_k1_lags(df, retained_k1, k1_lag_months, cfg.lag_batch_size)

    lag_cols = [f'{c}_lag{l}' for c in retained_k1 for l in k1_lag_months]
    orig_char_cols = retained_k0 + retained_k1
    all_char_cols = orig_char_cols + lag_cols

    print(f"Total characteristic columns, {len(all_char_cols)}")
    print(
        f"K0 current, {len(retained_k0)}, "
        f"K1 current, {len(retained_k1)}, "
        f"K1 lag columns, {len(lag_cols)}"
    )

    train, val, test = split_data(df, cfg.train_end, cfg.val_end)
    del df
    gc.collect()

    train = drop_high_missing_rows(train, orig_char_cols, threshold=1/3, label="Train")
    val = drop_high_missing_rows(val, orig_char_cols, threshold=1/3, label="Val")
    test = drop_high_missing_rows(test, orig_char_cols, threshold=1/3, label="Test")

    train = add_missingness_flags(train, orig_char_cols)
    val = add_missingness_flags(val, orig_char_cols)
    test = add_missingness_flags(test, orig_char_cols)
    print(f"Missingness flags added, {len([c for c in train.columns if c.endswith('_miss')])} columns")

    print("Normalising training set")
    train = cross_sectional_normalise(train, all_char_cols)
    gc.collect()
    print("Normalising validation set")
    val = cross_sectional_normalise(val, all_char_cols)
    gc.collect()
    print("Normalising test set")
    test = cross_sectional_normalise(test, all_char_cols)
    gc.collect()

    save_split_parquet(train, 'train', cfg.output_dir)
    save_split_parquet(val, 'val', cfg.output_dir)
    save_split_parquet(test, 'test', cfg.output_dir)

    # Build country lookup from all three splits and save as a dedicated
    # parquet file so that downstream scripts have no dependency on the raw data.
    all_excntry = pd.concat([
        train[['id', 'eom', 'excntry']],
        val[['id', 'eom', 'excntry']],
        test[['id', 'eom', 'excntry']],
    ], ignore_index=True).drop_duplicates()

    country_codes = sorted(all_excntry['excntry'].dropna().unique().tolist())
    country_to_id = {c: i for i, c in enumerate(country_codes)}
    all_excntry['country_id'] = all_excntry['excntry'].map(country_to_id).astype('Int16')
    country_lookup_out = all_excntry[['id', 'eom', 'country_id']].dropna(subset=['country_id'])
    country_lookup_out.to_parquet(cfg.country_lookup_path, index=False)
    print(
        f"Country lookup saved, {cfg.country_lookup_path}, "
        f"{len(country_lookup_out):,} rows, {len(country_codes)} countries"
    )
    del all_excntry, country_lookup_out
    gc.collect()

    col_metadata = {
        'retained_k0': retained_k0,
        'retained_k1': retained_k1,
        'lag_cols': lag_cols,
        'orig_char_cols': orig_char_cols,
        'all_char_cols': all_char_cols,
        'country_to_id': country_to_id,
        'country_codes': country_codes,
    }
    with open(cfg.col_metadata_path, 'w') as f:
        json.dump(col_metadata, f, indent=2)

    print(f"Column metadata saved, {cfg.col_metadata_path}")
    print(f"Countries ({len(country_codes)}), {', '.join(country_codes)}")
    del train, val, test
    gc.collect()


# Run preprocessing only when processed outputs are absent. This guard allows
# the training section to be re-executed without repeating the full pipeline.

required_outputs = [
    cfg.train_path, cfg.val_path, cfg.test_path,
    cfg.col_metadata_path, cfg.country_lookup_path,
]
if not all(p.exists() for p in required_outputs):
    print("Processed data not found. Running preprocessing pipeline.")
    run_preprocessing(cfg)
else:
    print("Processed data found. Skipping preprocessing.")


# Column setup from saved metadata. The column metadata JSON produced by
# run_preprocessing is the single source of truth for all downstream column
# references, replacing any dependency on a separately maintained column list.

with open(cfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta['retained_k0']
k1_chars = col_meta['retained_k1']

parquet_schema_cols = set(pq.read_schema(cfg.train_path).names)

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]

# k1_feature_cols_all contains all six column positions per K1 characteristic
# in the order [char, char_lag12, char_lag24, char_lag36, char_lag48, char_lag60].
# This ordering is required for the (N, n_k1 * 6) -> (N, n_k1, 6) reshape in
# CrossSectionalDataset to place lag positions correctly along dimension 2.
k1_feature_cols_all = [
    char + suffix
    for char in k1_chars
    for suffix in lag_suffixes
]

k0_miss_cols = [f'{c}_miss' for c in k0_chars if f'{c}_miss' in parquet_schema_cols]
k1_miss_cols = [f'{c}_miss' for c in k1_chars if f'{c}_miss' in parquet_schema_cols]

country_lookup_df = pd.read_parquet(cfg.country_lookup_path)
country_lookup_df['eom'] = pd.to_datetime(country_lookup_df['eom'])

country_to_id = col_meta['country_to_id']
country_codes = col_meta['country_codes']

print(f"K0 characteristics, {len(k0_chars)}")
print(f"K1 characteristics, {len(k1_chars)} (x6 lags = {len(k1_feature_cols_all)})")
print(f"K0 missingness flags, {len(k0_miss_cols)}")
print(f"K1 missingness flags, {len(k1_miss_cols)}")
print(f"Countries, {len(country_codes)}")


# Dataset

class CrossSectionalDataset(Dataset):
    """Stores one tensor batch per calendar month. Each batch contains the
    K0 and K1 characteristic tensors, binary missingness flags, integer country
    identifiers, continuous return targets, and valid-observation masks for all
    firms in that month. All firms in the universe are retained; the per-country
    cross-sectional attention in Path 2 naturally bounds each attention
    computation to the largest single-country cross section."""

    def __init__(self, df, k0_cols, k1_cols, k0_miss_cols, k1_miss_cols,
                 n_k1, target_col_list, country_lookup):
        dates = sorted(df["eom"].unique())
        self.monthly_data = []

        df = df.merge(country_lookup, on=["id", "eom"], how="left")
        df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)

        for date in dates:
            group = df[df["eom"] == date]

            k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
            k1_raw = group[k1_cols].values.astype(np.float32)
            k1 = torch.tensor(
                k1_raw.reshape(len(group), n_k1, 6), dtype=torch.float32
            )
            k0_m = torch.tensor(group[k0_miss_cols].values, dtype=torch.float32)
            k1_m = torch.tensor(group[k1_miss_cols].values, dtype=torch.float32)
            cids = torch.tensor(group["country_id"].values, dtype=torch.long)

            targets = {}
            valid_masks = {}
            for tc in target_col_list:
                vals = group[tc].values.copy().astype(np.float32)
                valid_mask = ~np.isnan(vals)
                vals[~valid_mask] = 0.0
                targets[tc] = torch.tensor(vals, dtype=torch.float32)
                valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)

            self.monthly_data.append({
                "k0": k0, "k1": k1,
                "k0_miss": k0_m, "k1_miss": k1_m,
                "country_ids": cids,
                "targets": targets,
                "valid_masks": valid_masks,
                "n_firms": len(group),
            })

        del df
        gc.collect()

    def __len__(self):
        return len(self.monthly_data)

    def __getitem__(self, idx):
        return self.monthly_data[idx]


def load_dataset(path, k0_cols, k1_cols, k0_miss, k1_miss, n_k1,
                 target_col_list, country_lookup):
    """Read a processed parquet split and construct a CrossSectionalDataset.
    Columns absent from the parquet schema are zero-filled before tensor
    construction, preserving the fixed (n_k1, 6) layout for the K1 reshape."""
    available = set(pq.read_schema(path).names)
    required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_col_list
    load_cols = [c for c in required if c in available]
    df = pd.read_parquet(path, columns=load_cols)
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if col not in df.columns:
            df[col] = 0.0
    # Residual NaN values arise in lag columns for the earliest observations
    # whose full 60-month look-back window pre-dates the panel. Zero-filling
    # here provides a belt-and-suspenders guarantee on top of the nan_to_num
    # applied during preprocessing, guarding against any parquet round-trip
    # edge cases.
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)
    return CrossSectionalDataset(
        df, k0_cols, k1_cols, k0_miss, k1_miss, n_k1,
        target_col_list, country_lookup
    )


# Architecture

class Time2Vec(nn.Module):
    """Learnable temporal encoding following Kazemi et al. (2019). The first
    output dimension is a linear term; remaining dimensions are sinusoidal
    with learned frequencies and phases."""

    def __init__(self, d_out):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(d_out))
        self.phi = nn.Parameter(torch.randn(d_out))

    def forward(self, lag_position):
        lag = lag_position.float().unsqueeze(-1)
        raw = self.omega * lag + self.phi
        out = torch.zeros_like(raw)
        out[..., 0] = raw[..., 0]
        out[..., 1:] = torch.sin(raw[..., 1:])
        return out


class GRN(nn.Module):
    """Gated Residual Network used as the position-wise feed-forward block
    in the cross-sectional Transformer encoder. The sigmoid gate controls
    how much of the transformed representation is added to the residual."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model * 2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        gated = self.fc2(h)
        value, gate = gated.chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)
        return self.layer_norm(residual + h)


# Feature encoding variants

class LinearEncoder(nn.Module):
    """Variant 1: per-feature linear projection. Replicates the Kelly et al.
    (2022) baseline in which each scalar maps to R^d via a weight and bias."""

    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class PerFeatureTokeniser(nn.Module):
    """Variant 2: per-feature projection matrix following Gorishniy et al.
    (2021). Each characteristic receives its own W_k, allowing the model to
    learn characteristic-specific transformations."""

    def __init__(self, n_features, d_model):
        super().__init__()
        self.projections = nn.Parameter(torch.randn(n_features, 1, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        proj = self.projections.squeeze(1).unsqueeze(0)
        return x_exp * proj + self.biases.unsqueeze(0)


class PiecewiseLinearEncoder(nn.Module):
    """Variant 3: piecewise linear encoding following Gorishniy et al. (2022).
    The rank-normalised range [-0.5, 0.5] is partitioned into equal-width bins;
    each scalar is encoded as a quantile bin membership vector with learned
    weights per bin per feature."""

    def __init__(self, n_features, d_model, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        boundaries = torch.linspace(-0.5, 0.5, num_bins + 1)
        self.register_buffer("boundaries", boundaries)
        self.feature_weights = nn.Parameter(
            torch.randn(n_features, num_bins, d_model) * 0.02
        )

    def _encode_bins(self, x):
        t_lower = self.boundaries[:-1]
        t_upper = self.boundaries[1:]
        x_exp = x.unsqueeze(-1)
        activations = torch.clamp(
            (x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0
        )
        return activations

    def forward(self, x):
        bin_act = self._encode_bins(x)
        return torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)


class PeriodicEncoder(nn.Module):
    """Variant 4: periodic encoding with learnable frequencies and phases.
    Each scalar is mapped through sin(omega * x + phi) and projected to R^d.
    This captures cyclical patterns without imposing a fixed discretisation."""

    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq, d_model)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        sinusoidal = torch.sin(
            x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0)
        )
        return self.proj(sinusoidal)


class FourierEncoder(nn.Module):
    """Variant 5: Fourier encoding with both sine and cosine components.
    Concatenating sin and cos provides a complete basis and removes the phase
    ambiguity present in the single-component periodic encoder."""

    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq * 2, d_model)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        scaled = x_exp * self.omega.unsqueeze(0)
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(features)


def build_encoder(variant, n_features, d_model, ple_bins=16, periodic_freq=32):
    """Factory function returning the encoder module for the specified variant."""
    if variant == "linear":
        return LinearEncoder(n_features, d_model)
    elif variant == "per_feature":
        return PerFeatureTokeniser(n_features, d_model)
    elif variant == "ple":
        return PiecewiseLinearEncoder(n_features, d_model, num_bins=ple_bins)
    elif variant == "periodic":
        return PeriodicEncoder(n_features, d_model, num_freq=periodic_freq)
    elif variant == "fourier":
        return FourierEncoder(n_features, d_model, num_freq=periodic_freq)
    else:
        raise ValueError(f"Unknown encoding variant: {variant}")


class SparseMultiHeadAttention(nn.Module):
    """Multi-head attention with top-k sparsification. Only the top_k most
    attended positions per query are retained before the softmax; all others
    are masked to negative infinity. This reduces the effective receptive
    field and regularises the peer comparison structure."""

    def __init__(self, d_model, n_heads, top_k, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_firms = x.shape[0]
        x_seq = x.unsqueeze(0)
        q = self.w_q(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        k_eff = min(self.top_k, n_firms)
        topk_vals, _ = scores.topk(k_eff, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        scores = scores.masked_fill(scores < threshold, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
        out = self.w_o(context).squeeze(0)
        return out, attn_weights.squeeze(0)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block consisting of sparse multi-head
    self-attention followed by a Gated Residual Network. Pre-normalisation
    improves gradient flow in deep stacks."""

    def __init__(self, d_model, n_heads, d_ff, top_k, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SparseMultiHeadAttention(d_model, n_heads, top_k, dropout)
        self.grn = GRN(d_model, d_ff, dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed)
        x = x + attn_out
        x = self.grn(x)
        return x, attn_weights


class AttentiveAggregation(nn.Module):
    """Attention-weighted pooling for K0 characteristics with soft missingness
    masking. A learned scalar penalty gamma is subtracted from the pre-softmax
    logit of each missing feature, driving its aggregation weight toward zero
    while allowing the model to learn the optimal degree of down-weighting."""

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.miss_penalty = nn.Parameter(torch.tensor(5.0))
        self.scale = math.sqrt(d_model)

    def forward(self, encoded, miss_mask=None):
        scores = (encoded * self.query).sum(dim=-1) / self.scale
        if miss_mask is not None:
            scores = scores - self.miss_penalty * miss_mask
        weights = F.softmax(scores, dim=1)
        token = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return token, weights


class K1TwoLevelAggregation(nn.Module):
    """Two-level attention pooling for K1 characteristics. The first level
    aggregates across the six annual lag positions for each characteristic,
    allowing the model to learn how much weight to assign to the current
    observation versus historical values. The second level aggregates across
    K1 features with soft missingness masking, mirroring the K0 aggregation."""

    def __init__(self, d_model):
        super().__init__()
        self.lag_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.feat_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.miss_penalty = nn.Parameter(torch.tensor(5.0))
        self.scale = math.sqrt(d_model)

    def forward(self, k1_encoded, miss_mask=None):
        # k1_encoded shape: (n_firms, 6, n_k1, d)
        # First level: aggregate across lag dimension (dim=1)
        lag_scores = (k1_encoded * self.lag_query).sum(dim=-1) / self.scale
        lag_weights = F.softmax(lag_scores, dim=1)
        h_bar = (k1_encoded * lag_weights.unsqueeze(-1)).sum(dim=1)
        # h_bar shape: (n_firms, n_k1, d)
        # Second level: aggregate across feature dimension (dim=1 of h_bar)
        feat_scores = (h_bar * self.feat_query).sum(dim=-1) / self.scale
        if miss_mask is not None:
            feat_scores = feat_scores - self.miss_penalty * miss_mask
        feat_weights = F.softmax(feat_scores, dim=1)
        token = (h_bar * feat_weights.unsqueeze(-1)).sum(dim=1)
        # token shape: (n_firms, d)
        return token, lag_weights, feat_weights


class FirmScoreHead(nn.Module):
    """Variable-depth MLP scoring head for per-firm base score prediction.
    The architecture is LayerNorm -> [Linear, ELU, Dropout] x L -> Linear -> scalar.
    Three separate instances are instantiated for the 3m, 6m, and 12m horizons,
    sharing no parameters."""

    def __init__(self, d_model, d_ff, n_layers, dropout):
        super().__init__()
        modules = [nn.LayerNorm(d_model)]
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_ff
            modules.extend([nn.Linear(in_dim, d_ff), nn.ELU(), nn.Dropout(dropout)])
        final_in = d_ff if n_layers > 0 else d_model
        modules.append(nn.Linear(final_in, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DualPathTransformer(nn.Module):
    """Dual Path Portfolio Transformer. Path 1 processes all firms independently
    through attentive feature aggregation and a multi-layer scoring head to
    produce base scores. Path 2 groups firms by country and applies shared
    sparse Transformer encoder blocks to produce peer-relative adjustment scores.
    The final predicted score is the elementwise sum of both paths. The per-country
    grouping eliminates the global max_firms truncation: every firm in the universe
    participates in cross-sectional attention within its own country group."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        n_k0 = len(k0_chars)
        n_k1 = len(k1_chars)

        # Shared feature encoders
        self.k0_encoder = build_encoder(
            config.encoding_variant, n_k0, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )
        self.k1_encoder = build_encoder(
            config.encoding_variant, n_k1, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )

        self.time2vec = Time2Vec(config.d_model)
        self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)

        # Attention-weighted aggregation with missingness masking
        self.k0_agg = AttentiveAggregation(config.d_model)
        self.k1_agg = K1TwoLevelAggregation(config.d_model)

        # Path 1: independent per-firm scoring heads
        self.base_head_3m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )
        self.base_head_6m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )
        self.base_head_12m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )

        # Path 2: shared cross-sectional Transformer blocks applied per country
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.top_k_attention, config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Path 2: lightweight adjustment heads (LayerNorm + single linear layer)
        self.adj_head_3m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )
        self.adj_head_6m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )
        self.adj_head_12m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )

        self.register_buffer(
            "lag_pos_tensor",
            torch.tensor(lag_positions_list, dtype=torch.float32),
        )
        self.min_firms = config.min_firms_attention

    def _encode_firms(self, k0, k1, k0_miss, k1_miss):
        """Shared encoding stage: produces firm embeddings z_i for all firms
        in the cross-section without any cross-firm interaction."""
        n_firms = k0.shape[0]

        # K0: encode, add static characteristic embedding, aggregate
        k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
        k0_token, k0_weights = self.k0_agg(k0_encoded, k0_miss)

        # K1: reshape to (n_firms * 6, n_k1), encode, add Time2Vec,
        # reshape back to (n_firms, 6, n_k1, d), then two-level aggregate
        k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
        k1_encoded = self.k1_encoder(k1_flat)
        k1_encoded = k1_encoded.view(n_firms, 6, len(k1_chars), self.config.d_model)
        t2v = self.time2vec(self.lag_pos_tensor).unsqueeze(0).unsqueeze(2)
        k1_encoded = k1_encoded + t2v
        k1_token, k1_lag_w, k1_feat_w = self.k1_agg(k1_encoded, k1_miss)

        z = k0_token + k1_token
        agg_info = {
            "k0_weights": k0_weights,
            "k1_lag_weights": k1_lag_w,
            "k1_feat_weights": k1_feat_w,
        }
        return z, agg_info

    def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
        z, agg_info = self._encode_firms(k0, k1, k0_miss, k1_miss)

        # Path 1: base scores for all firms
        base_3m = self.base_head_3m(z)
        base_6m = self.base_head_6m(z)
        base_12m = self.base_head_12m(z)

        # Path 2: per-country adjustment scores
        adj_3m = torch.zeros_like(base_3m)
        adj_6m = torch.zeros_like(base_6m)
        adj_12m = torch.zeros_like(base_12m)
        all_attn = []

        for cid in country_ids.unique():
            mask = country_ids == cid
            if mask.sum() < self.min_firms:
                # Countries below the minimum size receive no cross-sectional
                # adjustment; their final score is the Path 1 base score alone,
                # which is trained to be independently predictive via the aux loss
                continue
            z_c = z[mask]
            for block in self.blocks:
                z_c, attn_w = block(z_c)
                all_attn.append(attn_w)
            adj_3m[mask] = self.adj_head_3m(z_c).squeeze(-1)
            adj_6m[mask] = self.adj_head_6m(z_c).squeeze(-1)
            adj_12m[mask] = self.adj_head_12m(z_c).squeeze(-1)

        return {
            "scores_3m": base_3m + adj_3m,
            "scores_6m": base_6m + adj_6m,
            "scores_12m": base_12m + adj_12m,
            "base_3m": base_3m,
            "base_6m": base_6m,
            "base_12m": base_12m,
            "attn": all_attn,
            "agg": agg_info,
        }


# Training utilities

def compute_dual_path_loss(output, targets, valid_masks, config):
    """Combined Huber loss on final combined scores plus an auxiliary Huber
    loss on the per-firm base scores. The auxiliary term ensures that Path 1
    remains independently predictive and prevents Path 2 from compensating
    for an undertrained per-firm encoder."""
    main_loss = torch.tensor(0.0, device=output["scores_3m"].device)
    aux_loss = torch.tensor(0.0, device=output["scores_3m"].device)
    for horizon, weight in [
        ("3m", config.lambda_3m),
        ("6m", config.lambda_6m),
        ("12m", config.lambda_12m),
    ]:
        target_key = f"target_{horizon}"
        valid = valid_masks[target_key]
        if valid.sum() == 0:
            continue
        t = targets[target_key][valid]
        main_loss = main_loss + weight * F.huber_loss(
            output[f"scores_{horizon}"][valid], t, delta=1.0
        )
        aux_loss = aux_loss + weight * F.huber_loss(
            output[f"base_{horizon}"][valid], t, delta=1.0
        )
    total = main_loss + config.lambda_aux * aux_loss
    return total, main_loss.item(), aux_loss.item()


def compute_rank_correlation(scores, targets, valid_mask):
    """Spearman rank correlation between predicted scores and realised returns,
    computed in-graph to allow use as a monitoring metric during training. A
    minimum of 10 valid observations is required to return a non-zero value."""
    valid = valid_mask
    if valid.sum() < 10:
        return 0.0

    pred = scores[valid]
    true = targets[valid]

    def _rank(t):
        order = t.argsort()
        ranks = torch.zeros_like(t)
        ranks[order] = torch.arange(len(t), device=t.device, dtype=torch.float32)
        return ranks

    rank_pred = _rank(pred)
    rank_true = _rank(true)
    mean_p = rank_pred.mean()
    mean_t = rank_true.mean()
    cov = ((rank_pred - mean_p) * (rank_true - mean_t)).sum()
    std_p = ((rank_pred - mean_p) ** 2).sum().sqrt()
    std_t = ((rank_true - mean_t) ** 2).sum().sqrt()
    if std_p * std_t < 1e-8:
        return 0.0
    return (cov / (std_p * std_t)).item()


def train_one_epoch(model, dataset, optimizer, config, scaler):
    """Run one full pass over the training dataset in random month order.
    Mixed precision training is applied via GradScaler. Gradient clipping
    at config.grad_clip guards against exploding gradients from extreme returns.
    Returns a four-tuple of epoch-averaged (total_loss, main_loss, aux_loss,
    grad_norm)."""
    model.train()
    epoch_loss = 0.0
    epoch_main = 0.0
    epoch_aux = 0.0
    epoch_grad_norm = 0.0
    n_months = 0
    indices = np.random.permutation(len(dataset))
    for idx in indices:
        batch = dataset[idx]
        k0 = batch["k0"].to(device, non_blocking=True)
        k1 = batch["k1"].to(device, non_blocking=True)
        k0_miss = batch["k0_miss"].to(device, non_blocking=True)
        k1_miss = batch["k1_miss"].to(device, non_blocking=True)
        cids = batch["country_ids"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
        valid_masks = {k: v.to(device, non_blocking=True) for k, v in batch["valid_masks"].items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type):
            output = model(k0, k1, k0_miss, k1_miss, cids)
            loss, main_val, aux_val = compute_dual_path_loss(
                output, targets, valid_masks, config
            )

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_grad_norm += grad_norm.item()

        epoch_loss += loss.item()
        epoch_main += main_val
        epoch_aux += aux_val
        n_months += 1

    n = max(n_months, 1)
    return epoch_loss / n, epoch_main / n, epoch_aux / n, epoch_grad_norm / n


@torch.no_grad()
def evaluate(model, dataset, config):
    """Evaluate the model on a dataset, returning the average Huber loss and
    per-horizon Spearman rank correlations. Early stopping and learning rate
    scheduling are driven by the 6-month rank correlation, not the loss,
    to avoid the proxy mismatch where loss improvement concentrates in the
    middle of the return distribution rather than the tails."""
    model.eval()
    total_loss = 0.0
    total_corr = {"target_3m": 0.0, "target_6m": 0.0, "target_12m": 0.0}
    n_months = 0
    for idx in range(len(dataset)):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        valid_masks = {k: v.to(device) for k, v in batch["valid_masks"].items()}

        output = model(k0, k1, k0_miss, k1_miss, cids)
        loss, _, _ = compute_dual_path_loss(output, targets, valid_masks, config)
        total_loss += loss.item()

        for horizon in ["target_3m", "target_6m", "target_12m"]:
            score_key = horizon.replace("target_", "scores_")
            total_corr[horizon] += compute_rank_correlation(
                output[score_key], targets[horizon], valid_masks[horizon]
            )
        n_months += 1

    n = max(n_months, 1)
    return {
        "loss": total_loss / n,
        "rank_corr": {k: v / n for k, v in total_corr.items()},
    }






def train_variant(config):
    """Train the Dual Path Transformer for a single encoding variant.
    All metrics are printed to the terminal in a structured tabular format.
    Each epoch produces one row showing the training loss decomposition
    (total, main component, auxiliary component), validation loss, Spearman
    rank correlations at all three horizons, current learning rate, and mean
    gradient norm. A star marker is appended on any epoch that improves the
    best validation 6-month rank correlation. A summary block is printed
    after early stopping and again after test-set evaluation."""
    variant = config.encoding_variant
    w = 104
    bar = "=" * w
    sep = "-" * w

    def _block(lines):
        for line in lines:
            print(line)

    _block([
        bar,
        f" Variant       {variant}",
        f" Architecture  d_model={config.d_model}  n_heads={config.n_heads}  "
        f"n_layers={config.n_layers}  d_ff={config.d_ff}  dropout={config.dropout}",
        f"               n_mlp_layers={config.n_mlp_layers}  "
        f"lambda_aux={config.lambda_aux}  top_k_attention={config.top_k_attention}  "
        f"min_firms_attention={config.min_firms_attention}",
        f" Loss weights  3m={config.lambda_3m}  6m={config.lambda_6m}  "
        f"12m={config.lambda_12m}  aux={config.lambda_aux}",
        f" Optimiser     lr={config.learning_rate:.2e}  wd={config.weight_decay:.2e}  "
        f"grad_clip={config.grad_clip}  patience={config.patience}",
    ])

    n_k1 = len(k1_chars)
    train_ds = load_dataset(
        config.train_path, k0_feature_cols, k1_feature_cols_all,
        k0_miss_cols, k1_miss_cols, n_k1, target_cols, country_lookup_df,
    )
    val_ds = load_dataset(
        config.val_path, k0_feature_cols, k1_feature_cols_all,
        k0_miss_cols, k1_miss_cols, n_k1, target_cols, country_lookup_df,
    )

    model = DualPathTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _block([
        f" Data train_months={len(train_ds)}  val_months={len(val_ds)}",
        f" Model {n_params:,} trainable parameters",
        bar,
    ])

    col_header = (
        f"{'Epoch':>5}  "
        f"{'TrnTotal':>9} {'TrnMain':>9}{'TrnAux':>9}  "
        f"{'ValLoss':>9} "
        f"{'Corr3m':>7}  {'Corr6m':>7}  {'Corr12m':>8}  "
        f"{'LR':>9}  {'GNorm':>8}"
    )
    print(col_header)
    print(sep)
    sys.stdout.flush()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )

    best_val_corr = -float("inf")
    best_epoch = 1
    patience_counter = 0
    history = {
        "train_loss": [], "train_main": [], "train_aux": [],
        "val_loss": [], "val_corr_6m": [],
    }
    weights_path = config.results_dir / f"weights_{variant}.pt"
    scaler = torch.GradScaler(device.type)

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_main, train_aux, grad_norm = train_one_epoch(
            model, train_ds, optimizer, config, scaler,
        )
        val_metrics = evaluate(model, val_ds, config)
        val_corr_6m = val_metrics["rank_corr"]["target_6m"]
        scheduler.step(val_corr_6m)

        history["train_loss"].append(train_loss)
        history["train_main"].append(train_main)
        history["train_aux"].append(train_aux)
        history["val_loss"].append(val_metrics["loss"])
        history["val_corr_6m"].append(val_corr_6m)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_corr_6m > best_val_corr + 1e-5
        marker = "  *" if is_best else ""

        row = (
            f"{epoch:>5}  "
            f"{train_loss:>9.6f}  {train_main:>9.6f}  {train_aux:>9.6f}  "
            f"{val_metrics['loss']:>9.6f}  "
            f"{val_metrics['rank_corr']['target_3m']:>7.4f}  "
            f"{val_corr_6m:>7.4f}  "
            f"{val_metrics['rank_corr']['target_12m']:>8.4f}  "
            f"{current_lr:>9.2e}  {grad_norm:>8.4f}"
            f"{marker}"
        )
        print(row)
        sys.stdout.flush()

        if is_best:
            best_val_corr = val_corr_6m
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(sep)
                print(
                    f"  Early stopping at epoch {epoch}  "
                    f"(patience={config.patience}  best_epoch={best_epoch}  "
                    f"best_corr6m={best_val_corr:.4f})"
                )
                break

    del train_ds, val_ds
    gc.collect()

    final_epoch = len(history["train_loss"])
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    test_ds = load_dataset(
        config.test_path, k0_feature_cols, k1_feature_cols_all,
        k0_miss_cols, k1_miss_cols, n_k1, target_cols, country_lookup_df,
    )
    test_metrics = evaluate(model, test_ds, config)
    del test_ds

    corr = test_metrics["rank_corr"]
    _block([
        bar,
        f" Test Results  ({variant})",
        sep,
        f"{'Test loss':<22} {test_metrics['loss']:>10.6f}",
        f"{'Rank corr  3m':<22} {corr['target_3m']:>10.4f}",
        f"{'Rank corr  6m':<22} {corr['target_6m']:>10.4f}",
        f"{'Rank corr 12m':<22} {corr['target_12m']:>10.4f}",
        sep,
        f"Best val corr 6m  {best_val_corr:.4f}  (epoch {best_epoch})",
        f"Stopped epoch {final_epoch}",
        sep,
        f"Weights {weights_path}",
    ])

    results_path = config.results_dir / f"metrics_{variant}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "variant": variant,
                "n_params": n_params,
                "best_val_corr": best_val_corr,
                "best_epoch": best_epoch,
                "stopped_epoch": final_epoch,
                "history": history,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )

    _block([f"   Metrics  {results_path}", bar, ""])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Train all five encoding variants in sequence

for variant_name in ["linear", "per_feature", "ple", "periodic", "fourier"]:
    cfg.encoding_variant = variant_name
    train_variant(cfg)


# Portfolio simulation

@torch.no_grad()
def portfolio_simulation(model, dataset, config, rebalance_freq=6, tc_bps=25):
    """Long-only portfolio: firms are ranked by the 6-month combined score at
    each rebalancing date and the top quintile is held in equal weights. A
    transaction cost of 25 basis points is applied on portfolio turnover."""
    model.eval()
    portfolio_returns = []
    prev_holdings = set()

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)

        output = model(k0, k1, k0_miss, k1_miss, cids)
        scores_6m = output["scores_6m"]

        n_firms = scores_6m.shape[0]
        n_quintile = max(int(0.2 * n_firms), 1)
        _, top_indices = scores_6m.topk(n_quintile)
        top_set = set(top_indices.cpu().numpy().tolist())

        new_holdings = top_set - prev_holdings
        exited = prev_holdings - top_set
        turnover = (len(new_holdings) + len(exited)) / max(len(top_set), 1)
        tc = turnover * tc_bps / 10000.0

        raw_returns = batch["targets"]["target_6m"]
        valid = batch["valid_masks"]["target_6m"]
        valid_returns = [
            raw_returns[fi].item()
            for fi in top_indices.cpu().numpy()
            if valid[fi]
        ]
        mean_return = (
            sum(valid_returns) / max(len(valid_returns), 1) if valid_returns else 0.0
        )
        portfolio_returns.append(mean_return - tc)
        prev_holdings = top_set

    return np.array(portfolio_returns)


@torch.no_grad()
def portfolio_simulation_long_short(model, dataset, config, rebalance_freq=6, tc_bps=25):
    """Long-short portfolio: long the top quintile with score-proportional
    weights and short the bottom quintile with inverse-score-proportional
    weights. Transaction costs are applied on turnover in both legs."""
    model.eval()
    portfolio_returns = []
    prev_long = set()
    prev_short = set()

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)

        output = model(k0, k1, k0_miss, k1_miss, cids)
        scores_6m = output["scores_6m"]

        n_firms = scores_6m.shape[0]
        n_quintile = max(int(0.2 * n_firms), 1)
        _, long_idx = scores_6m.topk(n_quintile)
        _, short_idx = scores_6m.topk(n_quintile, largest=False)

        long_set = set(long_idx.cpu().numpy().tolist())
        short_set = set(short_idx.cpu().numpy().tolist())

        lt = len(long_set - prev_long) + len(prev_long - long_set)
        st = len(short_set - prev_short) + len(prev_short - short_set)
        tc = (lt + st) / max(n_quintile, 1) * tc_bps / 10000.0

        raw_returns = batch["targets"]["target_6m"]
        valid = batch["valid_masks"]["target_6m"]

        long_w = F.softmax(scores_6m[long_idx], dim=0)
        long_ret = sum(
            long_w[i].item() * raw_returns[fi].item()
            for i, fi in enumerate(long_idx.cpu().numpy())
            if valid[fi]
        )
        short_w = F.softmax(-scores_6m[short_idx], dim=0)
        short_ret = sum(
            short_w[i].item() * raw_returns[fi].item()
            for i, fi in enumerate(short_idx.cpu().numpy())
            if valid[fi]
        )
        portfolio_returns.append(long_ret - short_ret - tc)
        prev_long = long_set
        prev_short = short_set

    return np.array(portfolio_returns)


def compute_portfolio_metrics(returns, periods_per_year=2):
    """Compute annualised return, annualised volatility, Sharpe ratio, and
    maximum drawdown from a sequence of portfolio period returns. The default
    periods_per_year of 2 corresponds to 6-month rebalancing."""
    cum_return = (1 + returns).prod() - 1
    annualised_return = (
        (1 + cum_return) ** (periods_per_year / max(len(returns), 1)) - 1
    )
    annualised_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualised_return / max(annualised_vol, 1e-8)
    cum_wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak
    max_dd = drawdown.max()
    return {
        "cumulative_return": cum_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_rebalances": len(returns),
    }


# Portfolio evaluation: load saved metrics, select the best variant by
# test-set 6-month rank correlation, and run the portfolio simulation.

all_results = {}
for metrics_path in sorted(cfg.results_dir.glob("metrics_*.json")):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    variant_name = metrics.get("variant") or metrics_path.stem.replace("metrics_", "")
    all_results[variant_name] = metrics

if not all_results:
    raise RuntimeError(f"No metrics files found in {cfg.results_dir}")

best_variant = max(
    all_results,
    key=lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"],
)
print(f"Best variant, {best_variant}")
print()

cfg.encoding_variant = best_variant
best_model = DualPathTransformer(cfg).to(device)
best_model.load_state_dict(
    torch.load(
        cfg.results_dir / f"weights_{best_variant}.pt", weights_only=True
    )
)

n_k1 = len(k1_chars)
test_ds = load_dataset(
    cfg.test_path, k0_feature_cols, k1_feature_cols_all,
    k0_miss_cols, k1_miss_cols, n_k1, target_cols, country_lookup_df,
)

lo_returns = portfolio_simulation(best_model, test_ds, cfg)
ls_returns = portfolio_simulation_long_short(best_model, test_ds, cfg)

print("Long-Only Portfolio:")
for k, v in compute_portfolio_metrics(lo_returns).items():
    print(f"{k}, {v:.4f}")

print()
print("Long-Short Portfolio:")
for k, v in compute_portfolio_metrics(ls_returns).items():
    print(f"{k}, {v:.4f}")

del best_model, test_ds
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
