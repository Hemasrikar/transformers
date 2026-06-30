import gc
import json
import time
import pickle
import warnings
from pathlib import Path

import pyarrow as pa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from safetensors.torch import save_file as safetensors_save
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print(f'cuda available: {cuda_available}, device: {device}')
if cuda_available:
    print(f'device name: {torch.cuda.get_device_name(0)}')


# configuration

data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/mlp_benchmark')
results_dir.mkdir(parents=True, exist_ok=True)

train_end = pd.Timestamp('2015-12-31')
val_end = pd.Timestamp('2020-12-31')

ret_col_1m = 'ret_exc_lead1m'
ret_col = 'ret_exc_lead6m'

rebalance_freq = 6
horizon_months = 6
tc_bps = 25
min_stocks = 30
ret_clip_low = -1.0
ret_clip_high = 1.0

target_vol = 0.10
vol_lookback_months = 36
max_leverage_long_only = 3.0
max_leverage_long_short = 3.0
max_position_weight = 0.05

n_epochs_hpo = 100
patience = 10
grad_clip_norm = 1.0

n_trials = 30
optuna_seed = 24
torch_seed = 24

periods_per_year = 12.0 / rebalance_freq
n_vol_periods = max(1, vol_lookback_months // rebalance_freq)



schema = pq.read_schema(data_path)

non_feature = {
    # identifiers
    'id', 'gvkey', 'iid', 'isin', 'cusip', 'permno', 'permco',
    # dates, country, currency, size grouping
    'eom', 'date', 'excntry', 'curcd', 'size_grp',
    # the prediction target column at the one month horizon, retained here so
    # the cumulative six month target can be constructed below
    ret_col_1m,
    # industry classification codes encoded as float
    'sic', 'naics', 'gics', 'ff49',
    # exchange and share classification codes
    'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
    # filter and quality indicators, all encoded as float
    'obs_main', 'exch_main', 'primary_sec', 'common', 'bidask',
    'source_crsp',
    # return calculation metadata
    'adjfct', 'fx', 'ret_lag_dif',
    # raw same period returns, redundant with ret_1_0 short term reversal characteristic
    'ret', 'ret_exc', 'ret_local',
    # level forms of characteristics, redundant with the ranked characteristics
    'me', 'me_company', 'prc', 'prc_local', 'prc_high', 'prc_low',
    'dolvol', 'shares', 'tvol',
}
feature_cols = [
    c for c in schema.names
    if c not in non_feature
    and pa.types.is_floating(schema.field(c).type)
    and '_lag' not in c
]

print(f'feature columns selected: {len(feature_cols)}')

needed = list(dict.fromkeys(
    [c for c in ['id', 'eom', 'excntry', ret_col_1m] + feature_cols
     if c in schema.names]
))

df = pd.read_parquet(data_path, columns=needed)
df['eom'] = pd.to_datetime(df['eom'])

for col in feature_cols:
    if col in df.columns and df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)

df[ret_col_1m] = df[ret_col_1m].clip(lower=ret_clip_low, upper=ret_clip_high)

print(f'loaded: {df.shape[0]:,} rows, {len(feature_cols)} characteristic columns')
print(f'date range: {df["eom"].min().date()} to {df["eom"].max().date()}')


# six month cumulative forward target. for each firm and month we compound
# the next six one month forward returns. the block must be complete. any
# firm month with a gap in the forward window is dropped for that month.

df = df.sort_values(['id', 'eom']).reset_index(drop=True)

shifted = []
for k in range(horizon_months):
    s = df.groupby('id', sort=False)[ret_col_1m].shift(-k)
    shifted.append(s.to_numpy(dtype=np.float64))

shifted = np.stack(shifted, axis=1)
valid_block = np.isfinite(shifted).all(axis=1)

cum = np.where(
    valid_block,
    np.prod(1.0 + shifted, axis=1) - 1.0,
    np.nan,
)
df[ret_col] = cum.astype(np.float32)
df[ret_col] = df[ret_col].clip(lower=ret_clip_low * 2.0, upper=ret_clip_high * 2.0)

retained = int(np.isfinite(cum).sum())
print(f'cumulative six month target: {retained:,} of {len(df):,} rows retained')
print(f'  retention rate: {100.0 * retained / len(df):.2f}%')

del shifted
gc.collect()


# per month preprocessing: rank normalise each characteristic to the unit
# interval, centre at zero, impute missing to zero (the cross sectional
# median after centering). matches tree_benchmark.py exactly.

sorted_eoms = sorted(df['eom'].unique())
all_months = {}
n_feat = len(feature_cols)

for eom in sorted_eoms:
    month = df[df['eom'] == eom].copy()
    month = month[month[ret_col].notna()]
    if len(month) < min_stocks:
        continue
    ids = month['id'].to_numpy()
    r = month[ret_col].to_numpy().astype(np.float64)
    x = np.zeros((len(month), n_feat), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        if col not in month.columns:
            continue
        vals = month[col].astype(np.float64).to_numpy()
        valid = np.isfinite(vals)
        if valid.sum() > 1:
            ranks = pd.Series(vals[valid]).rank(pct=True).to_numpy(dtype=np.float32)
            x[valid, j] = ranks - 0.5
    r1m = month[ret_col_1m].to_numpy().astype(np.float64)
    all_months[eom] = {'ids': ids, 'r': r, 'r1m': r1m, 'x': x}

sorted_dates = sorted(all_months.keys())
print(f'processed: {len(sorted_dates)} months')
print(f'avg firms per month: {np.mean([len(m["ids"]) for m in all_months.values()]):.0f}')


# train, validation, and test splits

train_dates = [d for d in sorted_dates if d <= train_end]
val_dates = [d for d in sorted_dates if train_end < d <= val_end]
test_dates = [d for d in sorted_dates if d > val_end]

x_train = np.vstack([all_months[d]['x'] for d in train_dates])
y_train = np.concatenate([all_months[d]['r'] for d in train_dates]).astype(np.float32)

print(f'train: {len(train_dates)} months, val: {len(val_dates)} months, test: {len(test_dates)} months')
print(f'x_train: {x_train.shape}')


# model

class MLP(nn.Module):
    def __init__(self, n_features, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPPredictor:
    def __init__(self, model, dev):
        self.model = model
        self.dev = dev

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().to(self.dev)
            return self.model(x_t).cpu().numpy()


# portfolio helpers. these are identical to tree_benchmark.py so that the
# two benchmarks share exactly the same simulation and metrics conventions.

def portfolio_metrics(rets, ppy, dates=None):
    rets = np.asarray(rets, dtype=np.float64)
    if len(rets) == 0:
        out = {
            'ann_ret': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan,
            'se_sharpe': np.nan, 'max_dd': np.nan, 'cum_return': np.nan,
            'n_obs': 0,
        }
        if dates is not None:
            out['per_year'] = {}
        return out
    n = len(rets)
    ann_ret = float(rets.mean() * ppy)
    ann_vol = float(rets.std() * np.sqrt(ppy))
    sharpe = ann_ret / max(ann_vol, 1e-8)
    se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n))
    cw = np.cumprod(1.0 + rets)
    pk = np.maximum.accumulate(cw)
    max_dd = float(((pk - cw) / pk).max()) if len(cw) > 0 else 0.0
    cum_return = float(cw[-1] - 1.0)

    out = {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'se_sharpe': se,
        'max_dd': max_dd,
        'cum_return': cum_return,
        'n_obs': n,
    }

    if dates is not None:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        per_year = {}
        for y in sorted(set(years.tolist())):
            mask = years == y
            sub = rets[mask]
            if len(sub) < 1:
                continue
            y_ret = float(sub.mean() * ppy)
            y_vol = float(sub.std() * np.sqrt(ppy))
            y_sharpe = y_ret / max(y_vol, 1e-8)
            ycw = np.cumprod(1.0 + sub)
            ypk = np.maximum.accumulate(ycw)
            y_dd = float(((ypk - ycw) / ypk).max())
            per_year[int(y)] = {
                'ann_ret': y_ret,
                'ann_vol': y_vol,
                'sharpe': y_sharpe,
                'max_dd': y_dd,
                'cum_return': float(ycw[-1] - 1.0),
                'n_obs': int(len(sub)),
            }
        out['per_year'] = per_year

    return out


def _capped_softmax_weights(scores, max_weight, max_iter=20):
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if max_weight <= 1.0 / n + 1e-12:
        return np.full(n, 1.0 / n, dtype=np.float64)
    z = scores - scores.max()
    w = np.exp(z)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = w / s
    for _ in range(max_iter):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        residual = ~over
        residual_total = float(w[residual].sum())
        if residual_total <= 1e-12:
            break
        w = np.where(over, max_weight, w)
        w = np.where(residual, w * (1.0 + excess / residual_total), w)
    return w


def _renorm_over_valid(weights, valid):
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return weights
    valid_total = float(weights[valid].sum())
    if valid_total <= 1e-12:
        return weights
    out = np.zeros_like(weights)
    out[valid] = weights[valid] / valid_total
    return out


def _firm_id_turnover(prev_ids, curr_ids):
    prev = set(prev_ids.tolist()) if prev_ids is not None else set()
    curr = set(curr_ids.tolist())
    if not curr:
        return 0.0
    return (len(curr - prev) + len(prev - curr)) / max(len(curr), 1)


def _weight_l1_turnover(prev_ids, prev_w, curr_ids, curr_w):
    if curr_ids is None or curr_w is None or len(curr_w) == 0:
        return 0.0
    curr_map = {}
    for j in range(len(curr_ids)):
        fid = int(curr_ids[j]) if not hasattr(curr_ids[j], 'item') else int(curr_ids[j].item())
        curr_map[fid] = float(curr_w[j])
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return float(sum(abs(v) for v in curr_map.values()))
    prev_map = {}
    for j in range(len(prev_ids)):
        fid = int(prev_ids[j]) if not hasattr(prev_ids[j], 'item') else int(prev_ids[j].item())
        prev_map[fid] = float(prev_w[j])
    all_ids = set(prev_map.keys()) | set(curr_map.keys())
    return float(sum(
        abs(curr_map.get(fid, 0.0) - prev_map.get(fid, 0.0)) for fid in all_ids
    ))


def _drift_weights(prev_ids, prev_w, realised_returns_by_id):
    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return None, None
    n = len(prev_w)
    ids_list = []
    growth = np.zeros(n, dtype=np.float64)
    for j in range(n):
        fid = int(prev_ids[j]) if not hasattr(prev_ids[j], 'item') else int(prev_ids[j].item())
        ids_list.append(fid)
        growth[j] = float(prev_w[j]) * (1.0 + float(realised_returns_by_id.get(fid, 0.0)))
    g_sum = float(growth.sum())
    if g_sum > 1e-12:
        drifted = growth / g_sum
    else:
        drifted = growth
    return ids_list, drifted


def apply_period_vol_overlay(period_rets, n_vol_pds, ppy, max_lev):
    period_rets = np.asarray(period_rets, dtype=np.float64)
    n = len(period_rets)
    leverage = np.ones(n, dtype=np.float64)
    for t in range(n):
        if t < n_vol_pds:
            continue
        trailing = period_rets[t - n_vol_pds:t]
        if len(trailing) < 2:
            continue
        realised_vol = float(trailing.std() * np.sqrt(ppy))
        lev = target_vol / max(realised_vol, 1e-8)
        leverage[t] = float(np.clip(lev, 1.0 / max_lev, max_lev))
    return leverage


def apply_overlay_and_costs(leg_gross_rets, leg_tc, n_vol_pds, ppy, max_lev):
    leg_gross_rets = np.asarray(leg_gross_rets, dtype=np.float64)
    leg_tc = np.asarray(leg_tc, dtype=np.float64)
    leverage_path = apply_period_vol_overlay(leg_gross_rets, n_vol_pds, ppy, max_lev)
    unscaled_net = leg_gross_rets - leg_tc
    scaled_net = leverage_path * leg_gross_rets - leverage_path * leg_tc
    return scaled_net, unscaled_net, leverage_path


def predict_at_dates(predictor, month_dates):
    """Per firm predictions across the given dates."""
    rows = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = predictor.predict(m['x'])
        for k in range(len(m['ids'])):
            rows.append({
                'eom': eom,
                'id': m['ids'][k],
                'prediction': float(pred[k]),
                'realised_return': float(m['r'][k]),
            })
    return pd.DataFrame(rows)


def run_mean_split_simulation(predictor, month_dates):
    ls_period_rets, ls_period_dates = [], []
    ls_tc_history = []
    lo_period_rets, lo_period_dates = [], []
    lo_tc_history = []

    # state for drift-based turnover accounting per leg
    prev_long_ids = None
    prev_long_w = None
    prev_long_realised = None
    prev_short_ids = None
    prev_short_w = None
    prev_short_realised = None
    prev_lo_ids = None
    prev_lo_w = None
    prev_lo_realised = None

    ls_holdings, lo_holdings = [], []
    rb_counter = -1

    for pos, eom in enumerate(month_dates):
        if pos % rebalance_freq != 0:
            continue
        if eom not in all_months:
            continue
        m = all_months[eom]
        ids = m['ids']
        r = m['r']
        x = m['x']

        n_firms = len(ids)
        if n_firms < min_stocks:
            continue

        pred = predictor.predict(x)
        valid_pred = np.isfinite(pred)
        if valid_pred.sum() < min_stocks:
            continue

        valid_ret = np.isfinite(r)
        valid = valid_pred & valid_ret
        rb_counter += 1

        mean_score = float(pred[valid_pred].mean())
        long_mask = (pred > mean_score) & valid_pred
        short_mask = (pred <= mean_score) & valid_pred
        long_idx = np.where(long_mask)[0]
        short_idx = np.where(short_mask)[0]
        long_firm_ids = ids[long_idx]
        short_firm_ids = ids[short_idx]

        long_w = _capped_softmax_weights(pred[long_idx] - mean_score, max_position_weight)
        short_w = _capped_softmax_weights(mean_score - pred[short_idx], max_position_weight)
        long_w = _renorm_over_valid(long_w, valid[long_idx])
        short_w = _renorm_over_valid(short_w, valid[short_idx])

        # compute leg returns and record per-firm realised returns for
        # the drift accounting at the next rebalance
        long_ids_list = long_firm_ids.tolist()
        short_ids_list = short_firm_ids.tolist()
        long_realised = {}
        short_realised = {}
        long_ret = 0.0
        for i, fi in enumerate(long_idx):
            ri = float(r[fi]) if valid[fi] else 0.0
            long_realised[int(ids[fi])] = ri
            long_ret += long_w[i] * ri
        short_ret = 0.0
        for i, fi in enumerate(short_idx):
            ri = float(r[fi]) if valid[fi] else 0.0
            short_realised[int(ids[fi])] = ri
            short_ret += short_w[i] * ri
        ls_ret = long_ret - short_ret

        # drift previous leg weights and compute L1 turnover
        d_long_ids, d_long_w = _drift_weights(prev_long_ids, prev_long_w, prev_long_realised)
        d_short_ids, d_short_w = _drift_weights(prev_short_ids, prev_short_w, prev_short_realised)
        lt = _weight_l1_turnover(d_long_ids, d_long_w, long_ids_list, long_w)
        st = _weight_l1_turnover(d_short_ids, d_short_w, short_ids_list, short_w)
        ls_flat_tc = (lt + st) * tc_bps / 10000.0

        ls_period_rets.append(ls_ret)
        ls_period_dates.append(eom)
        ls_tc_history.append(ls_flat_tc)
        prev_long_ids = long_ids_list
        prev_long_w = long_w
        prev_long_realised = long_realised
        prev_short_ids = short_ids_list
        prev_short_w = short_w
        prev_short_realised = short_realised

        lo_w = _capped_softmax_weights(pred[valid_pred], max_position_weight)
        lo_w_full = np.zeros(n_firms, dtype=np.float64)
        lo_w_full[valid_pred] = lo_w
        lo_w_full = _renorm_over_valid(lo_w_full, valid)
        lo_firm_ids = ids[valid_pred]
        lo_ids_list = lo_firm_ids.tolist()
        lo_realised = {}
        lo_ret = 0.0
        for fi in range(n_firms):
            ri = float(r[fi]) if valid[fi] else 0.0
            lo_realised[int(ids[fi])] = ri
            lo_ret += lo_w_full[fi] * ri

        d_lo_ids, d_lo_w = _drift_weights(prev_lo_ids, prev_lo_w, prev_lo_realised)
        # the long-only target weights need to be expressed over the same
        # id space as the drifted previous weights. lo_w is indexed over
        # valid_pred positions. expand to the full id list for the L1 calc.
        lo_turn = _weight_l1_turnover(d_lo_ids, d_lo_w, ids.tolist(), lo_w_full)
        lo_flat_tc = lo_turn * tc_bps / 10000.0

        lo_period_rets.append(lo_ret)
        lo_period_dates.append(eom)
        lo_tc_history.append(lo_flat_tc)
        prev_lo_ids = ids.tolist()
        prev_lo_w = lo_w_full
        prev_lo_realised = lo_realised

        for i, fi in enumerate(long_idx):
            ls_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
                'id': int(ids[fi]), 'weight': float(long_w[i]),
                'realised_return': float(r[fi]) if valid[fi] else float('nan'),
            })
        for i, fi in enumerate(short_idx):
            ls_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'short',
                'id': int(ids[fi]), 'weight': float(-short_w[i]),
                'realised_return': float(r[fi]) if valid[fi] else float('nan'),
            })
        valid_pred_idx = np.where(valid_pred)[0]
        for i, fi in enumerate(valid_pred_idx):
            lo_holdings.append({
                'rebalance_index': rb_counter, 'eom': eom, 'leg': 'long',
                'id': int(ids[fi]), 'weight': float(lo_w[i]),
                'realised_return': float(r[fi]) if valid[fi] else float('nan'),
            })

    return {
        'long_short': {
            'returns': np.array(ls_period_rets),
            'tc': np.array(ls_tc_history),
            'dates': ls_period_dates,
            'holdings_df': pd.DataFrame(ls_holdings),
        },
        'long_only': {
            'returns': np.array(lo_period_rets),
            'tc': np.array(lo_tc_history),
            'dates': lo_period_dates,
            'holdings_df': pd.DataFrame(lo_holdings),
        },
    }


def _build_period_rows(model_name, portfolio, scaling, rets, dates):
    rets = np.asarray(rets, dtype=np.float64)
    if len(rets) == 0:
        return []
    cw = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cw)
    dd = (peak - cw) / peak

    roll_window = 4
    rolling_sharpe = np.full(len(rets), np.nan)
    rolling_ret = np.full(len(rets), np.nan)
    for i in range(roll_window - 1, len(rets)):
        w = rets[i - roll_window + 1:i + 1]
        mu = float(w.mean() * periods_per_year)
        sigma = float(w.std() * np.sqrt(periods_per_year))
        rolling_ret[i] = mu
        if sigma > 1e-12:
            rolling_sharpe[i] = mu / sigma

    rows = []
    for i, eom in enumerate(dates):
        rows.append({
            'model': model_name,
            'portfolio': portfolio,
            'scaling': scaling,
            'eom': pd.Timestamp(eom).strftime('%Y-%m-%d'),
            'return': round(float(rets[i]), 6),
            'cumulative_wealth': round(float(cw[i]), 6),
            'drawdown': round(float(dd[i]), 6),
            'rolling_sharpe_4p': (
                None if np.isnan(rolling_sharpe[i])
                else round(float(rolling_sharpe[i]), 4)
            ),
            'rolling_ann_ret_4p': (
                None if np.isnan(rolling_ret[i])
                else round(float(rolling_ret[i]) * 100, 4)
            ),
        })
    return rows


def _drift_weight_dict(weight_dict, id_to_r1m):
    growth = {}
    for fid, w in weight_dict.items():
        r = id_to_r1m.get(fid, 0.0)
        growth[fid] = w * (1.0 + r)
    g_sum = sum(growth.values())
    if g_sum > 1e-12:
        return {fid: v / g_sum for fid, v in growth.items()}
    return growth


def _weighted_return_from_dict(weight_dict, id_to_r1m):
    ret = 0.0
    for fid, w in weight_dict.items():
        r = id_to_r1m.get(fid, 0.0)
        ret += w * r
    return ret


def run_mean_split_simulation_monthly(predictor, month_dates):
    ls_monthly_rets, ls_monthly_tc, ls_monthly_dates, ls_rb_indices = [], [], [], []
    lo_monthly_rets, lo_monthly_tc, lo_monthly_dates, lo_rb_indices = [], [], [], []

    long_weight_dict = {}
    short_weight_dict = {}
    lo_weight_dict = {}

    prev_long_ids = None
    prev_long_w = None
    prev_short_ids = None
    prev_short_w = None
    prev_lo_ids = None
    prev_lo_w = None

    for pos, eom in enumerate(month_dates):
        if eom not in all_months:
            continue
        m = all_months[eom]
        ids = m['ids']
        r1m = m['r1m']
        valid_r1m = np.isfinite(r1m)

        ls_tc_this = 0.0
        lo_tc_this = 0.0

        if pos % rebalance_freq == 0:
            x = m['x']
            pred = predictor.predict(x)
            valid_pred = np.isfinite(pred)

            if valid_pred.sum() >= min_stocks:
                mean_score = float(pred[valid_pred].mean())
                long_mask = (pred > mean_score) & valid_pred
                short_mask = (pred <= mean_score) & valid_pred
                long_idx = np.where(long_mask)[0]
                short_idx = np.where(short_mask)[0]

                new_long_ids = ids[long_idx]
                new_short_ids = ids[short_idx]
                new_lo_ids = ids[valid_pred]

                lw = _capped_softmax_weights(pred[long_idx] - mean_score, max_position_weight)
                sw = _capped_softmax_weights(mean_score - pred[short_idx], max_position_weight)
                low = _capped_softmax_weights(pred[valid_pred], max_position_weight)

                # snapshot drifted previous weights for L1 turnover before
                # overwriting with new targets
                prev_drifted_long_ids = (list(long_weight_dict.keys()) if long_weight_dict else None)
                prev_drifted_long_w = (list(long_weight_dict.values()) if long_weight_dict else None)
                prev_drifted_short_ids = (list(short_weight_dict.keys()) if short_weight_dict else None)
                prev_drifted_short_w = (list(short_weight_dict.values()) if short_weight_dict else None)
                prev_drifted_lo_ids = (list(lo_weight_dict.keys()) if lo_weight_dict else None)
                prev_drifted_lo_w = (list(lo_weight_dict.values()) if lo_weight_dict else None)

                new_long_ids_list = new_long_ids.tolist()
                new_short_ids_list = new_short_ids.tolist()
                new_lo_ids_list = new_lo_ids.tolist()

                lt = _weight_l1_turnover(prev_drifted_long_ids, prev_drifted_long_w, new_long_ids_list, lw)
                st = _weight_l1_turnover(prev_drifted_short_ids, prev_drifted_short_w, new_short_ids_list, sw)
                lo_turn = _weight_l1_turnover(prev_drifted_lo_ids, prev_drifted_lo_w, new_lo_ids_list, low)

                ls_tc_this = (lt + st) * tc_bps / 10000.0
                lo_tc_this = lo_turn * tc_bps / 10000.0

                long_weight_dict = dict(zip(new_long_ids_list, lw.tolist()))
                short_weight_dict = dict(zip(new_short_ids_list, sw.tolist()))
                lo_weight_dict = dict(zip(new_lo_ids_list, low.tolist()))

                ls_rb_indices.append(len(ls_monthly_rets))
                lo_rb_indices.append(len(lo_monthly_rets))

        if not long_weight_dict:
            continue

        id_to_r1m = {
            int(fid): float(r1m[k])
            for k, fid in enumerate(ids.tolist())
            if valid_r1m[k]
        }

        long_ret = _weighted_return_from_dict(long_weight_dict, id_to_r1m)
        short_ret = _weighted_return_from_dict(short_weight_dict, id_to_r1m)
        lo_ret = _weighted_return_from_dict(lo_weight_dict, id_to_r1m)

        ls_monthly_rets.append(long_ret - short_ret)
        ls_monthly_tc.append(ls_tc_this)
        ls_monthly_dates.append(eom)

        lo_monthly_rets.append(lo_ret)
        lo_monthly_tc.append(lo_tc_this)
        lo_monthly_dates.append(eom)

        # drift weights forward by this month's realised returns so the
        # next month uses buy-and-hold weights rather than the original
        # target weights
        long_weight_dict = _drift_weight_dict(long_weight_dict, id_to_r1m)
        short_weight_dict = _drift_weight_dict(short_weight_dict, id_to_r1m)
        lo_weight_dict = _drift_weight_dict(lo_weight_dict, id_to_r1m)

    return {
        'long_short': {
            'returns': np.array(ls_monthly_rets),
            'tc': np.array(ls_monthly_tc),
            'dates': ls_monthly_dates,
            'rb_indices': ls_rb_indices,
        },
        'long_only': {
            'returns': np.array(lo_monthly_rets),
            'tc': np.array(lo_monthly_tc),
            'dates': lo_monthly_dates,
            'rb_indices': lo_rb_indices,
        },
    }


def apply_vol_target_monthly(monthly_rets, rebalance_indices, lookback_months, max_lev):
    monthly_rets = np.asarray(monthly_rets, dtype=np.float64)
    n = len(monthly_rets)
    leverage = np.ones(n, dtype=np.float64)
    n_rb = len(rebalance_indices)
    for i in range(n_rb):
        rb_idx = rebalance_indices[i]
        start = max(0, rb_idx - lookback_months)
        trailing = monthly_rets[start:rb_idx]
        if len(trailing) < lookback_months:
            continue
        sigma_ann = float(trailing.std() * np.sqrt(12.0))
        lev = float(np.clip(
            target_vol / max(sigma_ann, 1e-8),
            1.0 / max_lev, max_lev,
        ))
        next_rb = rebalance_indices[i + 1] if i + 1 < n_rb else n
        leverage[rb_idx:next_rb] = lev
    return leverage


def _build_monthly_rows(model_name, portfolio, scaling, rets, dates):
    rets = np.asarray(rets, dtype=np.float64)
    if len(rets) == 0:
        return []
    cw = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cw)
    dd = (peak - cw) / peak

    rolling_sharpe = np.full(len(rets), np.nan)
    rolling_ret = np.full(len(rets), np.nan)
    for i in range(11, len(rets)):
        w = rets[i - 11:i + 1]
        mu = float(w.mean() * 12.0)
        sigma = float(w.std() * np.sqrt(12.0))
        rolling_ret[i] = mu
        if sigma > 1e-12:
            rolling_sharpe[i] = mu / sigma

    rows = []
    for i, eom in enumerate(dates):
        rows.append({
            'model': model_name,
            'portfolio': portfolio,
            'scaling': scaling,
            'eom': pd.Timestamp(eom).strftime('%Y-%m-%d'),
            'return': round(float(rets[i]), 6),
            'cumulative_wealth': round(float(cw[i]), 6),
            'drawdown': round(float(dd[i]), 6),
            'rolling_sharpe_12m': (
                None if np.isnan(rolling_sharpe[i])
                else round(float(rolling_sharpe[i]), 4)
            ),
            'rolling_ann_ret_12m': (
                None if np.isnan(rolling_ret[i])
                else round(float(rolling_ret[i]) * 100, 4)
            ),
        })
    return rows


def rank_correlation_oos(predictor, month_dates):
    corrs = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = predictor.predict(m['x'])
        valid = np.isfinite(pred) & np.isfinite(m['r'])
        if valid.sum() < 10:
            continue
        result = spearmanr(pred[valid], m['r'][valid])
        c = result.statistic                                          # pyright: ignore[reportAttributeAccessIssue]
        if not np.isnan(c):
            corrs.append(float(c))                              
    return float(np.mean(corrs)) if corrs else 0.0


# training

def train_mlp(params, x_pool, y_pool, val_dates_local, n_epochs, patience_val, dev, seed, early_stop=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = MLP(
        n_features=x_pool.shape[1],
        d_model=params['d_model'],
        dropout=params['dropout'],
    ).to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    criterion = nn.MSELoss()

    x_t = torch.from_numpy(x_pool).float().to(dev)
    y_t = torch.from_numpy(y_pool).float().to(dev)
    n_total = len(x_t)
    batch_size = params['batch_size']
    predictor = MLPPredictor(model, dev)

    best_rc = -np.inf
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    train_losses, val_rank_corrs = [], []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_total, device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_total, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(x_t[idx])
            loss = criterion(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if early_stop:
            val_rc = rank_correlation_oos(predictor, val_dates_local)
            val_rank_corrs.append(val_rc)
            if val_rc > best_rc:
                best_rc = val_rc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience_val:
                    break
        else:
            best_epoch = epoch

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        'train_losses': train_losses,
        'val_rank_corrs': val_rank_corrs,
        'best_epoch': best_epoch,
        'best_val_rc': float(best_rc) if early_stop else float('nan'),
        'n_epochs_run': len(train_losses),
    }


# hyperparameter search. the objective is the validation long short sharpe
# under the mean split capped softmax construction with the volatility
# overlay applied. this matches the construction used for all other benchmarks.

mlp_best_params_path = results_dir / 'mlp_best_params.json'
mlp_study_path = results_dir / 'mlp_optuna_study.pkl'
mlp_trials_path = results_dir / 'mlp_optuna_trials.csv'

if mlp_best_params_path.exists():
    with open(mlp_best_params_path) as fh:
        cached = json.load(fh)
    mlp_best = cached['best_params']
    mlp_best_value = cached['best_value']
    mlp_best_epoch = int(cached['best_epoch'])
    mlp_hpo_time = cached['hpo_time_seconds']
    if mlp_study_path.exists():
        with open(mlp_study_path, 'rb') as fh:
            mlp_study = pickle.load(fh)
    else:
        mlp_study = None
    print(f'mlp best params loaded from {mlp_best_params_path.name}')
    print(f'mlp best val ls sharpe: {mlp_best_value:.4f}, best epoch: {mlp_best_epoch}')
else:
    def mlp_objective(trial):
        params = {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        }
        model, log = train_mlp(
            params=params,
            x_pool=x_train,
            y_pool=y_train,
            val_dates_local=val_dates,
            n_epochs=n_epochs_hpo,
            patience_val=patience,
            dev=device,
            seed=torch_seed,
            early_stop=True,
        )
        predictor = MLPPredictor(model, device)
        sim = run_mean_split_simulation(predictor, val_dates)
        ls = sim['long_short']
        lo = sim['long_only']
        if len(ls['returns']) == 0:
            return -999.0
        ls_scaled, _, _ = apply_overlay_and_costs(
            ls['returns'], ls['tc'], n_vol_periods, periods_per_year, max_leverage_long_short)
        lo_scaled, _, _ = apply_overlay_and_costs(
            lo['returns'], lo['tc'], n_vol_periods, periods_per_year, max_leverage_long_only)
        ls_sharpe = portfolio_metrics(ls_scaled, periods_per_year).get('sharpe', -999.0)
        lo_sharpe = portfolio_metrics(lo_scaled, periods_per_year).get('sharpe', -999.0)
        trial.set_user_attr('best_epoch', int(log['best_epoch']))
        trial.set_user_attr('n_epochs_run', int(log['n_epochs_run']))
        trial.set_user_attr('best_val_rc', float(log['best_val_rc']))
        trial.set_user_attr('val_sharpe_long_only', float(lo_sharpe))
        return float(ls_sharpe)

    mlp_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )
    t0 = time.time()
    mlp_study.optimize(mlp_objective, n_trials=n_trials, show_progress_bar=True)
    mlp_hpo_time = time.time() - t0
    mlp_best = mlp_study.best_params
    mlp_best_value = float(mlp_study.best_value)
    mlp_best_epoch = int(mlp_study.best_trial.user_attrs.get('best_epoch', n_epochs_hpo - 1))

    with open(mlp_best_params_path, 'w') as fh:
        json.dump({
            'construction': 'mean_split_softmax_cap_6m',
            'best_params': mlp_best,
            'best_value': mlp_best_value,
            'best_epoch': mlp_best_epoch,
            'best_trial_number': int(mlp_study.best_trial.number),
            'best_trial_user_attrs': dict(mlp_study.best_trial.user_attrs),
            'n_trials_completed': sum(
                1 for t in mlp_study.trials if t.state.name == 'COMPLETE'
            ),
            'hpo_time_seconds': float(mlp_hpo_time),
        }, fh, indent=2, default=float)

    mlp_study.trials_dataframe().to_csv(mlp_trials_path, index=False)
    with open(mlp_study_path, 'wb') as fh:
        pickle.dump(mlp_study, fh)

    print(f'mlp best val ls sharpe: {mlp_best_value:.4f}')
    print(f'mlp best params: {mlp_best}')
    print(f'mlp best epoch: {mlp_best_epoch}')
    print(f'mlp hpo time: {mlp_hpo_time:.1f} s')


# final training on training data only. trains for best_epoch + 1 epochs
# with no early stopping. the validation set was used to select best_epoch
# during the hpo search

n_final_epochs = mlp_best_epoch + 1

t0 = time.time()
mlp_model, mlp_log = train_mlp(
    params=mlp_best, x_pool=x_train,
    y_pool=y_train, val_dates_local=None,
    n_epochs=n_final_epochs, patience_val=patience,
    dev=device, seed=torch_seed,
    early_stop=False,
)
mlp_train_time = time.time() - t0
mlp_predictor = MLPPredictor(mlp_model, device)
n_params = sum(p.numel() for p in mlp_model.parameters())

print(f'mlp final model trained in {mlp_train_time:.1f} s, {n_final_epochs} epochs')
print(f'parameter count: {n_params:,}')

safetensors_save(mlp_model.state_dict(), str(results_dir / 'mlp_weights.safetensors'))

with open(results_dir / 'mlp_train_log.json', 'w') as mlps:
    json.dump({
        'train_losses': mlp_log['train_losses'],
        'best_epoch_from_hpo': mlp_best_epoch,
        'n_final_epochs': n_final_epochs,
        'training_time_seconds': float(mlp_train_time),
        'parameter_count': int(n_params),
    }, mlps, indent=2, default=float)


mlp_rc_val = rank_correlation_oos(mlp_predictor, val_dates)
mlp_rc_test = rank_correlation_oos(mlp_predictor, test_dates)
print(f'mlp rank corr: val = {mlp_rc_val:.4f}, test = {mlp_rc_test:.4f}')

# test set evaluation. the simulation runs on all sorted_dates so the
# volatility overlay has full warm up history before the test window begins.
# the return series is then sliced to the test window before computing metrics

def evaluate_and_save(predictor, name):
    sim = run_mean_split_simulation(predictor, sorted_dates)
    ls = sim['long_short']
    lo = sim['long_only']

    ls_scaled_full, ls_unscaled_full, ls_lev = apply_overlay_and_costs(
        ls['returns'], ls['tc'], n_vol_periods, periods_per_year, max_leverage_long_short)
    lo_scaled_full, lo_unscaled_full, lo_lev = apply_overlay_and_costs(
        lo['returns'], lo['tc'], n_vol_periods, periods_per_year, max_leverage_long_only)

    test_set = set(test_dates)
    ls_mask = np.array([d in test_set for d in ls['dates']])
    lo_mask = np.array([d in test_set for d in lo['dates']])

    ls_unscaled_test = ls_unscaled_full[ls_mask]
    ls_scaled_test = ls_scaled_full[ls_mask]
    lo_unscaled_test = lo_unscaled_full[lo_mask]
    lo_scaled_test = lo_scaled_full[lo_mask]

    ls_dates_test = [d for d, m in zip(ls['dates'], ls_mask) if m]
    lo_dates_test = [d for d, m in zip(lo['dates'], lo_mask) if m]

    ls_ret_df = pd.DataFrame({
        'eom': ls_dates_test, 'return_unscaled': ls_unscaled_test,
        'return_scaled': ls_scaled_test, 'leverage': ls_lev[ls_mask],
    })
    lo_ret_df = pd.DataFrame({
        'eom': lo_dates_test, 'return_unscaled': lo_unscaled_test,
        'return_scaled': lo_scaled_test, 'leverage': lo_lev[lo_mask],
    })

    ls_hold_df = (ls['holdings_df'][ls['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop=True))
    lo_hold_df = (lo['holdings_df'][lo['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop=True))
    m_ls_unscaled = portfolio_metrics(ls_unscaled_test, periods_per_year, dates=ls_dates_test)
    m_ls_scaled = portfolio_metrics(ls_scaled_test, periods_per_year, dates=ls_dates_test)
    m_lo_unscaled = portfolio_metrics(lo_unscaled_test, periods_per_year, dates=lo_dates_test)
    m_lo_scaled = portfolio_metrics(lo_scaled_test, periods_per_year, dates=lo_dates_test)

    ls_ret_df.to_csv(results_dir / f'{name}_returns_long_short.csv', index=False)
    lo_ret_df.to_csv(results_dir / f'{name}_returns_long_only.csv', index=False)
    ls_hold_df.to_csv(results_dir / f'{name}_holdings_long_short.csv', index=False)
    lo_hold_df.to_csv(results_dir / f'{name}_holdings_long_only.csv', index=False)

    predict_at_dates(predictor, test_dates).to_csv(
        results_dir / f'{name}_test_predictions.csv', index=False,
    )

    return {
        'returns_ls_unscaled': ls_unscaled_test,
        'returns_ls_scaled': ls_scaled_test,
        'returns_lo_unscaled': lo_unscaled_test,
        'returns_lo_scaled': lo_scaled_test,
        'dates_ls': ls_dates_test,
        'dates_lo': lo_dates_test,
        'metrics': {
            'long_short_unscaled': m_ls_unscaled,
            'long_short_scaled': m_ls_scaled,
            'long_only_unscaled': m_lo_unscaled,
            'long_only_scaled': m_lo_scaled,
        },
    }


mlp_eval = evaluate_and_save(mlp_predictor, 'mlp')

mls = mlp_eval['metrics']['long_short_scaled']
mlo = mlp_eval['metrics']['long_only_scaled']
print(f'mlp long short scaled: sharpe = {mls["sharpe"]:.4f}, '
    f'ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
print(f'mlp long only scaled: sharpe = {mlo["sharpe"]:.4f}, '
    f'ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')

# summary json

def _strip_per_year(m):
    if not isinstance(m, dict):
        return m
    return {k: v for k, v in m.items() if k != 'per_year'}


summary = {
    'construction': 'mean_split_softmax_cap_6m',
    'target_column': ret_col,
    'n_features': len(feature_cols),
    'feature_cols': feature_cols,
    'architecture': {
        'name': 'three_layer_mlp',
        'n_hidden_layers': 3,
        'hidden_width': mlp_best['d_model'],
        'activation': 'elu',
        'dropout': mlp_best['dropout'],
        'parameter_count': int(n_params),
    },
    'split': {
        'train': {
            'start': str(train_dates[0].date()), 'end': str(train_dates[-1].date()),
            'n_months': len(train_dates), 'n_obs': int(x_train.shape[0]),
        },
        'val': {
            'start': str(val_dates[0].date()), 'end': str(val_dates[-1].date()),
            'n_months': len(val_dates),
        },
        'test': {
            'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()),
            'n_months': len(test_dates),
        },
    },
    'config': {
        'rebalance_freq': rebalance_freq,
        'horizon_months': horizon_months,
        'tc_bps': tc_bps,
        'min_stocks': min_stocks,
        'ret_clip': [ret_clip_low, ret_clip_high],
        'target_vol': target_vol,
        'vol_lookback_months': vol_lookback_months,
        'n_vol_periods': n_vol_periods,
        'periods_per_year': periods_per_year,
        'max_leverage_long_only': max_leverage_long_only,
        'max_leverage_long_short': max_leverage_long_short,
        'max_position_weight': max_position_weight,
        'n_epochs_hpo': n_epochs_hpo,
        'n_final_epochs': n_final_epochs,
        'patience': patience,
        'grad_clip_norm': grad_clip_norm,
        'optuna_seed': optuna_seed,
        'torch_seed': torch_seed,
        'n_trials': n_trials,
    },
    'mlp': {
        'best_params': mlp_best,
        'best_val_long_short_sharpe': float(mlp_best_value),
        'best_trial_number': (
            int(mlp_study.best_trial.number) if mlp_study is not None else None
        ),
        'n_trials_completed': (
            sum(1 for t in mlp_study.trials if t.state.name == 'COMPLETE')
            if mlp_study is not None else None
        ),
        'hpo_time_seconds': float(mlp_hpo_time),
        'final_training_time_seconds': float(mlp_train_time),
        'best_epoch_from_hpo': mlp_best_epoch,
        'n_final_epochs': n_final_epochs,
        'rc_val': float(mlp_rc_val),
        'rc_test': float(mlp_rc_test),
        'portfolio_metrics': {k: _strip_per_year(v) for k, v in mlp_eval['metrics'].items()},
    },
}

with open(results_dir / 'mlp_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2, default=float)
print('summary json saved')


# headline summary csv

def _round_or_none(x, ndigits):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return round(float(x), ndigits)


rows = []
for portfolio, scaling, key in [
    ('long_short', 'unscaled', 'long_short_unscaled'),
    ('long_short', 'scaled', 'long_short_scaled'),
    ('long_only', 'unscaled', 'long_only_unscaled'),
    ('long_only', 'scaled', 'long_only_scaled'),
]:
    m = mlp_eval['metrics'][key]
    rows.append({
        'model': 'mlp',
        'portfolio': portfolio,
        'scaling': scaling,
        'rc_test': round(mlp_rc_test, 4),
        'sharpe': _round_or_none(m['sharpe'], 4),
        'se': _round_or_none(m['se_sharpe'], 4),
        'ann_ret': _round_or_none(m['ann_ret'] * 100, 2),
        'ann_vol': _round_or_none(m['ann_vol'] * 100, 2),
        'cum_return': _round_or_none(m['cum_return'] * 100, 2),
        'max_dd': _round_or_none(m['max_dd'] * 100, 2),
        'n_obs': m['n_obs'],
    })

summary_table = pd.DataFrame(rows)
print('MLP Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index=False))
summary_table.to_csv(results_dir / 'mlp_summary.csv', index=False)
print('summary csv saved')

# per year breakdown csv
per_year_rows = []

def _flush_per_year(model_name, portfolio, scaling, metrics):
    py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
    for year in sorted(py.keys()):
        ym = py[year]
        per_year_rows.append({
            'model': model_name,
            'portfolio': portfolio,
            'scaling': scaling,
            'year': int(year),
            'ann_ret': round(float(ym['ann_ret']) * 100, 4),
            'ann_vol': round(float(ym['ann_vol']) * 100, 4),
            'sharpe': (
                round(float(ym['sharpe']), 4)
                if not (isinstance(ym['sharpe'], float) and np.isnan(ym['sharpe']))
                else None
            ),
            'max_dd': round(float(ym['max_dd']) * 100, 4),
            'cum_return': round(float(ym['cum_return']) * 100, 4),
            'n_obs': int(ym['n_obs']),
        })


_flush_per_year('mlp', 'long_short', 'unscaled', mlp_eval['metrics']['long_short_unscaled'])
_flush_per_year('mlp', 'long_short', 'scaled', mlp_eval['metrics']['long_short_scaled'])
_flush_per_year('mlp', 'long_only', 'unscaled', mlp_eval['metrics']['long_only_unscaled'])
_flush_per_year('mlp', 'long_only', 'scaled', mlp_eval['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'mlp_per_year_metrics.csv', index=False)
print(f'per year metrics saved, {len(per_year_df)} rows')


# per period metrics csv

period_rows = []
for portfolio, scaling, rets, dates in [
    ('long_short', 'unscaled', mlp_eval['returns_ls_unscaled'], mlp_eval['dates_ls']),
    ('long_short', 'scaled', mlp_eval['returns_ls_scaled'], mlp_eval['dates_ls']),
    ('long_only', 'unscaled', mlp_eval['returns_lo_unscaled'], mlp_eval['dates_lo']),
    ('long_only', 'scaled', mlp_eval['returns_lo_scaled'], mlp_eval['dates_lo']),
]:
    period_rows.extend(_build_period_rows('mlp', portfolio, scaling, rets, dates))

per_period_df = pd.DataFrame(period_rows)
per_period_df.to_csv(results_dir / 'mlp_per_period_metrics.csv', index=False)
print(f'per period metrics saved, {len(per_period_df)} rows')


# monthly simulation. the monthly variant runs on sorted_dates so the
# vol overlay has full warm up before the test window. the return series
# is then sliced to the test window for metrics and csv output.
mo_sim = run_mean_split_simulation_monthly(mlp_predictor, sorted_dates)
mo_ls = mo_sim['long_short']
mo_lo = mo_sim['long_only']

mo_ls_lev = apply_vol_target_monthly(mo_ls['returns'] - mo_ls['tc'], mo_ls['rb_indices'], vol_lookback_months, max_leverage_long_short)
mo_lo_lev = apply_vol_target_monthly(mo_lo['returns'] - mo_lo['tc'], mo_lo['rb_indices'], vol_lookback_months, max_leverage_long_only)

mo_ls_unscaled_full = mo_ls['returns'] - mo_ls['tc']
mo_ls_scaled_full = mo_ls_lev * mo_ls['returns'] - mo_ls_lev * mo_ls['tc']
mo_lo_unscaled_full = mo_lo['returns'] - mo_lo['tc']
mo_lo_scaled_full = mo_lo_lev * mo_lo['returns'] - mo_lo_lev * mo_lo['tc']

test_set = set(test_dates)
mo_ls_mask = np.array([d in test_set for d in mo_ls['dates']])
mo_lo_mask = np.array([d in test_set for d in mo_lo['dates']])

mo_ls_unscaled_test = mo_ls_unscaled_full[mo_ls_mask]
mo_ls_scaled_test = mo_ls_scaled_full[mo_ls_mask]
mo_lo_unscaled_test = mo_lo_unscaled_full[mo_lo_mask]
mo_lo_scaled_test = mo_lo_scaled_full[mo_lo_mask]
mo_ls_dates_test = [d for d, m in zip(mo_ls['dates'], mo_ls_mask) if m]
mo_lo_dates_test = [d for d, m in zip(mo_lo['dates'], mo_lo_mask) if m]

mo_ls_unscaled_m = portfolio_metrics(mo_ls_unscaled_test, 12.0, dates=mo_ls_dates_test)
mo_ls_scaled_m = portfolio_metrics(mo_ls_scaled_test, 12.0, dates=mo_ls_dates_test)
mo_lo_unscaled_m = portfolio_metrics(mo_lo_unscaled_test, 12.0, dates=mo_lo_dates_test)
mo_lo_scaled_m = portfolio_metrics(mo_lo_scaled_test, 12.0, dates=mo_lo_dates_test)

print(f'mlp monthly long short scaled: sharpe = {mo_ls_scaled_m["sharpe"]:.4f},'
    f'ann_ret = {mo_ls_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {mo_ls_scaled_m["ann_vol"] * 100:.2f}%'
)
print(
    f'mlp monthly long only scaled: sharpe = {mo_lo_scaled_m["sharpe"]:.4f},'
    f'ann_ret = {mo_lo_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {mo_lo_scaled_m["ann_vol"] * 100:.2f}%'
)

monthly_rows = []
for portfolio, scaling, rets, dates in [
    ('long_short', 'unscaled', mo_ls_unscaled_test, mo_ls_dates_test),
    ('long_short', 'scaled', mo_ls_scaled_test, mo_ls_dates_test),
    ('long_only', 'unscaled', mo_lo_unscaled_test, mo_lo_dates_test),
    ('long_only', 'scaled', mo_lo_scaled_test, mo_lo_dates_test),
]:
    monthly_rows.extend(_build_monthly_rows('mlp', portfolio, scaling, rets, dates))

per_month_df = pd.DataFrame(monthly_rows)
per_month_df.to_csv(results_dir / 'mlp_per_month_metrics.csv', index=False)
print(f'per month metrics saved, {len(per_month_df)} rows')


# plots
mlp_color = 'darkgreen'
xlabel_periods = f'Rebalance periods from start of test window ({rebalance_freq} months each)'

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(np.cumprod(1 + mlp_eval['returns_ls_scaled']), label='MLP, Scaled', color=mlp_color)
ax.plot(np.cumprod(1 + mlp_eval['returns_ls_unscaled']), label='MLP, Unscaled', color=mlp_color, linestyle='--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Scaled and Unscaled')
ax.legend(frameon=False)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + mlp_eval['returns_lo_scaled']), label='MLP, Scaled', color=mlp_color)
ax.plot(np.cumprod(1 + mlp_eval['returns_lo_unscaled']), label='MLP, Unscaled', color=mlp_color, linestyle='--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Scaled and Unscaled')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(mlp_log['train_losses'], color='steelblue')
axes[0].set_xlabel('Epoch (final training pass)')
axes[0].set_ylabel('MSE')
axes[0].set_title('Training Loss')
axes[0].grid(alpha=0.3)

if mlp_study is not None:
    mlp_vals = [t.value for t in mlp_study.trials if t.value is not None]
    axes[1].plot(np.maximum.accumulate(mlp_vals), color=mlp_color)
    axes[1].scatter(range(len(mlp_vals)), mlp_vals, alpha=0.3, s=15, color=mlp_color)
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Validation LS Sharpe')
    axes[1].set_title('MLP Optuna Search')
    axes[1].grid(alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'study not in memory', ha='center', va='center')
    axes[1].set_title('MLP Optuna Search')

ls_unscaled = mlp_eval['returns_ls_unscaled']
ls_scaled = mlp_eval['returns_ls_scaled']
axes[2].bar(range(len(ls_unscaled)), ls_unscaled, alpha=0.5, color=mlp_color, label='Unscaled')
axes[2].bar(range(len(ls_scaled)), ls_scaled, alpha=0.5, color='darkorange', label='Scaled')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_xlabel(xlabel_periods)
axes[2].set_ylabel('Period Return')
axes[2].set_title('Long Short Period Returns')
axes[2].legend(frameon=False)
axes[2].grid(alpha=0.3)

fig.tight_layout()
plt.show()
