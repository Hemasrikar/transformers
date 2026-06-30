
# XGBoost and LightGBM Benchmark

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
import xgboost as xgb
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')



try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
except ImportError:
    torch = None
    cuda_available = False
    cuda_device_name = None

print(f'cuda available (torch): {cuda_available}')
if cuda_available:
    print(f'cuda device: {cuda_device_name}')


def _empty_cuda_cache():
    if cuda_available and torch is not None:
        torch.cuda.empty_cache()


def _gpu_cleanup():
    gc.collect()
    _empty_cuda_cache()


def _probe(probe_fn, label):
    try:
        probe_x = np.random.randn(200, 10).astype(np.float32)
        probe_y = np.random.randn(200).astype(np.float32)
        probe_fn(probe_x, probe_y)
        return True
    except Exception as exc:
        print(f'{label}: gpu probe failed ({type(exc).__name__}: {exc})')
        return False


xgb_use_cuda = False
lgb_use_gpu = False
if cuda_available:
    xgb_use_cuda = _probe(
        lambda x, y: xgb.XGBRegressor(n_estimators = 5, tree_method = 'hist', device = 'cuda', verbosity = 0).fit(x, y), 'xgboost',
    )
    lgb_use_gpu = _probe(
        lambda x, y: lgb.LGBMRegressor(n_estimators = 5, device = 'gpu', verbose = -1).fit(x, y), 'lightgbm',
    )

xgb_device_params = {'tree_method': 'hist', 'device': 'cuda'} if xgb_use_cuda else {'tree_method': 'hist'}
lgb_device_params = {'device': 'gpu'} if lgb_use_gpu else {}



data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/tree_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

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

n_trials_xgb = 50
n_trials_lgb = 50
optuna_seed = 24
n_hpo_months = 36



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

df = pd.read_parquet(data_path, columns = needed)
df['eom'] = pd.to_datetime(df['eom'])

for col in feature_cols:
    if col in df.columns and df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)

df[ret_col_1m] = df[ret_col_1m].clip(lower = ret_clip_low, upper = ret_clip_high)

print(f'loaded: {df.shape[0]:,} rows, {len(feature_cols)} characteristic columns')
print(f'date range: {df["eom"].min().date()} to {df["eom"].max().date()}')
print(f'countries: {df["excntry"].nunique()}')



# Construct the cumulative six month forward target. For each firm and
# month we form the product of one plus the next six one month forward
# returns, minus one. We require all six constituent observations to be
# present. Firms whose forward block contains a gap, namely a delisting or
# a missing return, are dropped from that month's cross section.

df = df.sort_values(['id', 'eom']).reset_index(drop = True)

# group by firm and shift the one month forward return backward by k months
# for k in 0.5, then compound. Shifting in this direction aligns
# ret_exc_lead1m at month t+k onto row t, the return realised
# between t+k and t+k+1, which is exactly the kth component of the six
# month forward block starting at t.

shifted = []
for k in range(horizon_months):
    s = df.groupby('id', sort = False)[ret_col_1m].shift(-k)
    shifted.append(s.to_numpy(dtype = np.float64))

shifted = np.stack(shifted, axis = 1)
valid_block = np.isfinite(shifted).all(axis = 1)

cum = np.where(
    valid_block,
    np.prod(1.0 + shifted, axis = 1) - 1.0,
    np.nan,
)
df[ret_col] = cum.astype(np.float32)

# clip the cumulative target to the same band as the underlying one month
# returns to avoid extreme outliers driving the loss. the band is wider
# than for one month returns because six month compounded returns have
# fatter tails
df[ret_col] = df[ret_col].clip(lower = ret_clip_low * 2.0, upper = ret_clip_high * 2.0)

retained = int(np.isfinite(cum).sum())
print(f'cumulative six month target constructed')
print(f'retained rows with valid six month forward block: {retained:,} of {len(df):,}')
print(f'retention: {100.0 * retained / len(df):.2f}%')

# drop the one month forward target from feature pool consideration. it
# was retained only to construct the six month target
del shifted
gc.collect()



# Per month preprocessing. For every cross section we rank normalise each
# characteristic to the unit interval, centre by subtracting 0.5 so that
# the cross sectional mean is approximately zero, and then impute the
# remaining missing values to zero. The imputation follows the benchmark
# methodology, under which the cross sectional median maps to 0.5 before
# centering and to zero after centering, so that imputed values do not
# affect the mean of any feature within the cross section.

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

    x = np.zeros((len(month), n_feat), dtype = np.float32)
    for j, col in enumerate(feature_cols):
        if col not in month.columns:
            continue
        vals = month[col].astype(np.float64).to_numpy()
        valid = np.isfinite(vals)
        if valid.sum() > 1:
            ranks = pd.Series(vals[valid]).rank(pct = True).to_numpy(dtype = np.float32)
            x[valid, j] = ranks - 0.5

    all_months[eom] = {'ids': ids, 'r': r, 'x': x}

sorted_dates = sorted(all_months.keys())
print(f'processed: {len(sorted_dates)} months')
print(f'avg firms/month: {np.mean([len(m["ids"]) for m in all_months.values()]):.0f}')


train_dates = [d for d in sorted_dates if d <= train_end]
val_dates = [d for d in sorted_dates if train_end < d <= val_end]
test_dates = [d for d in sorted_dates if d > val_end]

print(f'train: {len(train_dates)} months')
print(f'val: {len(val_dates)} months')
print(f'test: {len(test_dates)} months')

x_train = np.vstack([all_months[d]['x'] for d in train_dates])
y_train = np.concatenate([all_months[d]['r'] for d in train_dates])
print(f'x_train: {x_train.shape}')

hpo_dates = train_dates[-n_hpo_months:]
x_hpo = np.vstack([all_months[d]['x'] for d in hpo_dates])
y_hpo = np.concatenate([all_months[d]['r'] for d in hpo_dates])
print(f'x_hpo: {x_hpo.shape}')

trainval_dates = train_dates + val_dates
x_trainval = np.vstack([all_months[d]['x'] for d in trainval_dates])
y_trainval = np.concatenate([all_months[d]['r'] for d in trainval_dates])
print(f'x_trainval: {x_trainval.shape}')


def portfolio_metrics(rets, periods_per_year, dates = None):
    rets = np.asarray(rets, dtype = np.float64)
    if len(rets) == 0:
        return {}
    n = len(rets)
    ann_ret = float(rets.mean() * periods_per_year)
    ann_vol = float(rets.std() * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = ann_ret / max(ann_vol, 1e-8)
    se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n))
    cw = np.cumprod(1.0 + rets)
    pk = np.maximum.accumulate(cw)
    max_dd = float(((pk - cw) / pk).max()) if len(cw) > 0 else 0.0
    cum_return = float(cw[-1] - 1.0)
    years_elapsed = n / periods_per_year
    cagr = float(cw[-1] ** (1.0 / years_elapsed) - 1.0) if cw[-1] > 0 and years_elapsed > 0 else float('nan')

    out = {
        'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'se_sharpe': se,
        'max_dd': max_dd, 'cagr': cagr,
        'cum_return': cum_return, 'n_obs': n,
    }

    if dates is not None:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        per_year = {}
        for y in sorted(set(years.tolist())):
            mask = years == y
            sub = rets[mask]
            if len(sub) < 1:
                continue
            y_ret = float(sub.mean() * periods_per_year)
            y_vol = float(sub.std() * np.sqrt(periods_per_year)) if len(sub) > 1 else 0.0
            y_sharpe = y_ret / max(y_vol, 1e-8) if y_vol > 1e-12 else float('nan')
            ycw = np.cumprod(1.0 + sub)
            ypk = np.maximum.accumulate(ycw)
            y_dd = float(((ypk - ycw) / ypk).max())
            per_year[int(y)] = {
                'ann_ret': y_ret, 'ann_vol': y_vol,
                'sharpe': y_sharpe, 'max_dd': y_dd,
                'cum_return': float(ycw[-1] - 1.0),
                'n_obs': int(len(sub))
            }
        out['per_year'] = per_year

    return out


def _capped_softmax_weights(scores, max_weight, max_iter = 20):
    scores = np.asarray(scores, dtype = np.float64)
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype = np.float64)
    if max_weight <= 1.0 / n + 1e-12:
        return np.full(n, 1.0 / n, dtype = np.float64)

    # standard softmax in a numerically stable form
    z = scores - scores.max()
    w = np.exp(z)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(n, 1.0 / n, dtype = np.float64)
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
    weights = np.asarray(weights, dtype = np.float64)
    valid = np.asarray(valid, dtype = bool)
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
    new_in = len(curr - prev)
    exited = len(prev - curr)
    return (new_in + exited) / max(len(curr), 1)


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


def _drift_weights_arr(prev_ids, prev_w, realised_returns_by_id):

    if prev_ids is None or prev_w is None or len(prev_w) == 0:
        return None, None
    ids_list = []
    growth = np.zeros(len(prev_w), dtype = np.float64)
    for j in range(len(prev_w)):
        fid = int(prev_ids[j]) if not hasattr(prev_ids[j], 'item') else int(prev_ids[j].item())
        ids_list.append(fid)
        growth[j] = float(prev_w[j]) * (1.0 + float(realised_returns_by_id.get(fid, 0.0)))
    g_sum = float(growth.sum())
    if g_sum > 1e-12:
        drifted = growth / g_sum
    else:
        drifted = growth
    return ids_list, drifted


def apply_period_vol_overlay(period_rets, target_vol, n_vol_periods, periods_per_year, max_leverage):

    period_rets = np.asarray(period_rets, dtype = np.float64)
    n = len(period_rets)
    leverage_path = np.ones(n, dtype = np.float64)
    for t in range(n):
        if t < n_vol_periods:
            continue
        trailing = period_rets[t - n_vol_periods:t]
        if len(trailing) < 2:
            continue
        realised_vol = float(trailing.std() * np.sqrt(periods_per_year))
        lev = target_vol / max(realised_vol, 1e-8)
        leverage_path[t] = float(np.clip(lev, 1.0 / max_leverage, max_leverage))
    return leverage_path


def predict_at_dates(model, month_dates):
    """Per firm predictions across the given dates, returned as a long
    DataFrame with columns eom, id, prediction, realised_return."""
    rows = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = model.predict(m['x'])
        for k in range(len(m['ids'])):
            rows.append({
                'eom': eom,
                'id': m['ids'][k],
                'prediction': float(pred[k]),
                'realised_return': float(m['r'][k]),
            })
    return pd.DataFrame(rows)


def run_mean_split_simulation(model, month_dates):
    n_months = len(month_dates)

    ls_period_rets, ls_period_dates = [], []
    ls_tc_history = []
    lo_period_rets, lo_period_dates = [], []
    lo_tc_history = []

    # state for drift-based L1 turnover accounting per leg
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

        pred = model.predict(x)
        valid_pred = np.isfinite(pred)
        if valid_pred.sum() < min_stocks:
            continue

        valid_ret = np.isfinite(r)
        valid = valid_pred & valid_ret
        rb_counter += 1

        # long short leg construction by mean split
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
        # drift accounting at the next rebalance
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

        # drift previous leg weights and compute L1 turnover against them
        d_long_ids, d_long_w = _drift_weights_arr(prev_long_ids, prev_long_w, prev_long_realised)
        d_short_ids, d_short_w = _drift_weights_arr(prev_short_ids, prev_short_w, prev_short_realised)
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

        # long only leg construction holding all firms
        lo_w = _capped_softmax_weights(pred[valid_pred], max_position_weight)
        lo_w_full = np.zeros(n_firms, dtype = np.float64)
        lo_w_full[valid_pred] = lo_w
        lo_w_full = _renorm_over_valid(lo_w_full, valid)
        lo_ids_list = ids.tolist()
        lo_realised = {}
        lo_ret = 0.0
        for fi in range(n_firms):
            ri = float(r[fi]) if valid[fi] else 0.0
            lo_realised[int(ids[fi])] = ri
            lo_ret += lo_w_full[fi] * ri

        d_lo_ids, d_lo_w = _drift_weights_arr(prev_lo_ids, prev_lo_w, prev_lo_realised)
        lo_turn = _weight_l1_turnover(d_lo_ids, d_lo_w, lo_ids_list, lo_w_full)
        lo_flat_tc = lo_turn * tc_bps / 10000.0

        lo_period_rets.append(lo_ret)
        lo_period_dates.append(eom)
        lo_tc_history.append(lo_flat_tc)
        prev_lo_ids = lo_ids_list
        prev_lo_w = lo_w_full
        prev_lo_realised = lo_realised

        # record holdings
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


def apply_overlay_and_costs(leg_unscaled_rets, leg_tc, n_vol_periods, periods_per_year, max_leverage):
    """Combine leg returns, transaction costs, and the period volatility
    overlay. Mirrors the DPPT pattern where leverage scales both the gross
    leg return and the transaction cost. Returns a tuple (scaled_net_rets, unscaled_net_rets, leverage_path)."""
    leg_unscaled_rets = np.asarray(leg_unscaled_rets, dtype = np.float64)
    leg_tc = np.asarray(leg_tc, dtype = np.float64)
    leverage_path = apply_period_vol_overlay(
        leg_unscaled_rets, target_vol, n_vol_periods, periods_per_year, max_leverage,
    )
    unscaled_net = leg_unscaled_rets - leg_tc
    scaled_net = leverage_path * leg_unscaled_rets - leverage_path * leg_tc
    return scaled_net, unscaled_net, leverage_path



## Hyperparameter search

periods_per_year = 12.0 / rebalance_freq
n_vol_periods = max(1, vol_lookback_months // rebalance_freq)


def _trial_oom(exc):
    s = str(exc).lower()
    return 'out of memory' in s or 'cudaerrormemoryallocation' in s


def _eval_hpo_sharpe(model):
    sim = run_mean_split_simulation(model, val_dates)
    ls = sim['long_short']
    lo = sim['long_only']
    if len(ls['returns']) == 0:
        return -999.0, -999.0
    ls_scaled, _, _ = apply_overlay_and_costs(
        ls['returns'], ls['tc'], n_vol_periods, periods_per_year, max_leverage_long_short,
    )
    lo_scaled, _, _ = apply_overlay_and_costs(
        lo['returns'], lo['tc'], n_vol_periods, periods_per_year, max_leverage_long_only,
    )
    ls_sharpe = portfolio_metrics(ls_scaled, periods_per_year).get('sharpe', -999.0)
    lo_sharpe = portfolio_metrics(lo_scaled, periods_per_year).get('sharpe', -999.0)
    return float(ls_sharpe), float(lo_sharpe)


# xgboost hyperparameter search

xgb_best_params_path = results_dir / 'xgb_best_params.json'
xgb_study_path = results_dir / 'xgb_optuna_study.pkl'
xgb_trials_path = results_dir / 'xgb_optuna_trials.csv'

if xgb_best_params_path.exists():
    with open(xgb_best_params_path) as fh:
        cached = json.load(fh)
    xgb_best = cached['best_params']
    xgb_best_value = cached['best_value']
    xgb_hpo_time = cached['hpo_time_seconds']
    if xgb_study_path.exists():
        with open(xgb_study_path, 'rb') as fh:
            xgb_study = pickle.load(fh)
    else:
        xgb_study = None
    print(f'XGBoost hyperparameters already tuned, loaded from {xgb_best_params_path.name}')
    print(f'XGBoost best val ls sharpe: {xgb_best_value:.4f}')
    print(f'XGBoost best params: {xgb_best}')
else:
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log = True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log = True),
            'random_state': optuna_seed,
            'n_jobs': -1,
            'verbosity': 0,
            **xgb_device_params,
        }
        model = xgb.XGBRegressor(**params)
        try:
            try:
                model.fit(x_hpo, y_hpo)
            except Exception as exc:
                if _trial_oom(exc):
                    del model
                    _gpu_cleanup()
                    raise optuna.exceptions.TrialPruned()
                raise
            ls_sharpe, lo_sharpe = _eval_hpo_sharpe(model)
            trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
            return ls_sharpe
        finally:
            del model                                                   #type: ignore
            _gpu_cleanup()

    xgb_study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
    )
    t0 = time.time()
    xgb_study.optimize(xgb_objective, n_trials = n_trials_xgb, show_progress_bar = True)
    xgb_hpo_time = time.time() - t0
    xgb_best = xgb_study.best_params
    xgb_best_value = float(xgb_study.best_value)

    with open(xgb_best_params_path, 'w') as fh:
        json.dump({
            'construction': 'mean_split_softmax_cap_6m',
            'best_params': xgb_best,
            'best_value': xgb_best_value,
            'best_trial_number': int(xgb_study.best_trial.number),
            'best_trial_user_attrs': dict(xgb_study.best_trial.user_attrs),
            'n_trials_completed': sum(1 for t in xgb_study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(xgb_hpo_time),
        }, fh, indent = 2, default = float)

    xgb_trials_df = xgb_study.trials_dataframe()
    xgb_trials_df.to_csv(xgb_trials_path, index = False)
    with open(xgb_study_path, 'wb') as fh:
        pickle.dump(xgb_study, fh)

    print(f'XGBoost best val ls sharpe: {xgb_best_value:.4f}')
    print(f'XGBoost best params: {xgb_best}')
    print(f'XGBoost hpo time: {xgb_hpo_time:.1f} s, {xgb_hpo_time / 60:.2f} min')


# lightgbm hyperparameter search

lgb_best_params_path = results_dir / 'lgb_best_params.json'
lgb_study_path = results_dir / 'lgb_optuna_study.pkl'
lgb_trials_path = results_dir / 'lgb_optuna_trials.csv'

if lgb_best_params_path.exists():
    with open(lgb_best_params_path) as fh:
        cached = json.load(fh)
    lgb_best = cached['best_params']
    lgb_best_value = cached['best_value']
    lgb_hpo_time = cached['hpo_time_seconds']
    if lgb_study_path.exists():
        with open(lgb_study_path, 'rb') as fh:
            lgb_study = pickle.load(fh)
    else:
        lgb_study = None
    print(f'LightGBM hyperparameters already tuned, loaded from {lgb_best_params_path.name}')
    print(f'LightGBM best val ls sharpe: {lgb_best_value:.4f}')
    print(f'LightGBM best params: {lgb_best}')
else:
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log = True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log = True),
            'random_state': optuna_seed,
            'n_jobs': -1,
            'verbose': -1,
            **lgb_device_params,
        }
        model = lgb.LGBMRegressor(**params)
        try:
            try:
                model.fit(x_hpo, y_hpo)
            except Exception as exc:
                if _trial_oom(exc):
                    del model
                    _gpu_cleanup()
                    raise optuna.exceptions.TrialPruned()
                raise
            ls_sharpe, lo_sharpe = _eval_hpo_sharpe(model)
            trial.set_user_attr('val_sharpe_long_only', lo_sharpe)
            return ls_sharpe
        finally:
            del model                                                    #type: ignore 
            _gpu_cleanup()

    lgb_study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
    )
    t0 = time.time()
    lgb_study.optimize(lgb_objective, n_trials = n_trials_lgb, show_progress_bar = True)
    lgb_hpo_time = time.time() - t0
    lgb_best = lgb_study.best_params
    lgb_best_value = float(lgb_study.best_value)

    with open(lgb_best_params_path, 'w') as fh:
        json.dump({
            'construction': 'mean_split_softmax_cap_6m',
            'best_params': lgb_best,
            'best_value': lgb_best_value,
            'best_trial_number': int(lgb_study.best_trial.number),
            'best_trial_user_attrs': dict(lgb_study.best_trial.user_attrs),
            'n_trials_completed': sum(1 for t in lgb_study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(lgb_hpo_time),
        }, fh, indent = 2, default = float)

    lgb_trials_df = lgb_study.trials_dataframe()
    lgb_trials_df.to_csv(lgb_trials_path, index = False)
    with open(lgb_study_path, 'wb') as fh:
        pickle.dump(lgb_study, fh)

    print(f'LightGBM best val ls sharpe: {lgb_best_value:.4f}')
    print(f'LightGBM best params: {lgb_best}')
    print(f'LightGBM hpo time: {lgb_hpo_time:.1f} s, {lgb_hpo_time / 60:.2f} min')



_gpu_cleanup()


class XGBPredictor:
    """Thin wrapper exposing a sklearn style predict, save_model, and
    feature_importances_ on top of a native xgboost Booster."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, x):
        if not isinstance(x, xgb.DMatrix):
            x = xgb.DMatrix(x)
        return self.booster.predict(x)

    def save_model(self, path):
        self.booster.save_model(path)

    @property
    def feature_importances_(self):
        score = self.booster.get_score(importance_type = 'gain')
        imp = np.zeros(len(feature_cols), dtype = np.float64)
        for k, v in score.items():
            try:
                idx = int(k.lstrip('f'))
                if 0 <= idx < len(imp):
                    imp[idx] = v
            except ValueError:
                continue
        return imp


class ChunkIter(xgb.DataIter):
    """Hand the training matrix to xgboost in chunks. xgboost stores the
    quantised form of each chunk on disk and discards the raw chunk
    after consumption, so the peak memory footprint is bounded by the
    chunk size rather than by the full matrix."""
    def __init__(self, x, y, chunk_size, cache_prefix):
        self.x = x
        self.y = y
        self.chunk_size = chunk_size
        self.n_chunks = (len(x) + chunk_size - 1) // chunk_size
        self.current = 0
        super().__init__(cache_prefix = cache_prefix)

    def next(self, input_data):
        if self.current >= self.n_chunks:
            return False
        start = self.current * self.chunk_size
        end = min(start + self.chunk_size, len(self.x))
        input_data(data = self.x[start:end], label = self.y[start:end])
        self.current += 1
        return True

    def reset(self):
        self.current = 0


# build the booster parameter dictionary. The CPU device assignment is
# deliberate and required by the external memory mode
xgb_final_params = dict(xgb_best)
xgb_final_params.update({
    'tree_method': 'hist',
    'device': 'cpu',
    'max_bin': 64,
    'objective': 'reg:squarederror',
    'seed': optuna_seed,
    'nthread': -1,
})

n_estimators = int(xgb_final_params.pop('n_estimators', 100))

cache_dir = results_dir / 'xgb_cache'
cache_dir.mkdir(parents = True, exist_ok = True)

print('building xgboost external memory training matrix (train + val)')
it = ChunkIter(
    x = x_trainval,
    y = y_trainval,
    chunk_size = 200000,
    cache_prefix = str(cache_dir / 'iter'),
)
dtrain = xgb.ExtMemQuantileDMatrix(it, max_bin = 64)
print(f'matrix built: {dtrain.num_row():,} rows, {dtrain.num_col()} columns')

del it
gc.collect()

t0 = time.time()
xgb_booster = xgb.train(
    params = xgb_final_params,
    dtrain = dtrain,
    num_boost_round = n_estimators,
    verbose_eval = False,
)
xgb_train_time = time.time() - t0
xgb_model = XGBPredictor(xgb_booster)

del dtrain
gc.collect()
print(f'XGBoost final model trained in {xgb_train_time:.1f} s')


_gpu_cleanup()


lgb_final_params = {**lgb_device_params, **lgb_best}
lgb_final_params['max_bin'] = 128

lgb_model = lgb.LGBMRegressor(
    **lgb_final_params,
    random_state = optuna_seed,
    bagging_seed = optuna_seed,
    feature_fraction_seed = optuna_seed,
    data_random_seed = optuna_seed,
    deterministic = True,
    force_row_wise = True,
    n_jobs = -1,
    verbose = -1,
)
t0 = time.time()
lgb_model.fit(x_trainval, y_trainval)
lgb_train_time = time.time() - t0
print(f'LightGBM final model trained in {lgb_train_time:.1f} s')


_gpu_cleanup()


xgb_model.save_model(str(results_dir / 'xgb_model.json'))
lgb_model.booster_.save_model(str(results_dir / 'lgb_model.txt'))
print('models saved in native formats')



def rank_correlation_oos(model, month_dates):
    corrs = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = model.predict(m['x'])
        valid = np.isfinite(pred) & np.isfinite(m['r'])
        if valid.sum() < 10:
            continue
        result = spearmanr(pred[valid], m['r'][valid])
        c = result.statistic                                  # pyright: ignore[reportAttributeAccessIssue]
        if not np.isnan(c):
            corrs.append(float(c))                      
    return float(np.mean(corrs)) if corrs else 0.0


xgb_rc_val = rank_correlation_oos(xgb_model, val_dates)
xgb_rc_test = rank_correlation_oos(xgb_model, test_dates)
lgb_rc_val = rank_correlation_oos(lgb_model, val_dates)
lgb_rc_test = rank_correlation_oos(lgb_model, test_dates)

print(f'XGBoost rank corr: val = {xgb_rc_val:.4f}, test = {xgb_rc_test:.4f}')
print(f'LightGBM rank corr: val = {lgb_rc_val:.4f}, test = {lgb_rc_test:.4f}')



def evaluate_and_save(model, name):
    sim = run_mean_split_simulation(model, sorted_dates)
    ls = sim['long_short']
    lo = sim['long_only']

    ls_scaled_full, ls_unscaled_full, ls_lev = apply_overlay_and_costs(
        ls['returns'], ls['tc'], n_vol_periods, periods_per_year, max_leverage_long_short,
    )
    lo_scaled_full, lo_unscaled_full, lo_lev = apply_overlay_and_costs(
        lo['returns'], lo['tc'], n_vol_periods, periods_per_year, max_leverage_long_only,
    )

    test_set = set(test_dates)
    ls_mask = np.array([d in test_set for d in ls['dates']])
    lo_mask = np.array([d in test_set for d in lo['dates']])

    ls_raw_test = ls_unscaled_full[ls_mask]
    ls_scaled_test = ls_scaled_full[ls_mask]
    lo_raw_test = lo_unscaled_full[lo_mask]
    lo_scaled_test = lo_scaled_full[lo_mask]

    # test window rebalance dates, used both for the returns dataframe and
    # the per year breakdown inside portfolio_metrics
    ls_dates_test = [d for d, m in zip(ls['dates'], ls_mask) if m]
    lo_dates_test = [d for d, m in zip(lo['dates'], lo_mask) if m]

    ls_ret_df = pd.DataFrame({
        'eom': ls_dates_test,
        'return_unscaled': ls_raw_test,
        'return_scaled': ls_scaled_test,
        'leverage': ls_lev[ls_mask],
    })
    lo_ret_df = pd.DataFrame({
        'eom': lo_dates_test,
        'return_unscaled': lo_raw_test,
        'return_scaled': lo_scaled_test,
        'leverage': lo_lev[lo_mask],
    })

    ls_hold_df = ls['holdings_df'][ls['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop = True)
    lo_hold_df = lo['holdings_df'][lo['holdings_df']['eom'].isin(test_set)].copy().reset_index(drop = True)

    m_ls_raw = portfolio_metrics(ls_raw_test, periods_per_year, dates = ls_dates_test)
    m_ls_scaled = portfolio_metrics(ls_scaled_test, periods_per_year, dates = ls_dates_test)
    m_lo_raw = portfolio_metrics(lo_raw_test, periods_per_year, dates = lo_dates_test)
    m_lo_scaled = portfolio_metrics(lo_scaled_test, periods_per_year, dates = lo_dates_test)

    ls_ret_df.to_csv(results_dir / f'{name}_returns_long_short.csv', index = False)
    lo_ret_df.to_csv(results_dir / f'{name}_returns_long_only.csv', index = False)
    ls_hold_df.to_csv(results_dir / f'{name}_holdings_long_short.csv', index = False)
    lo_hold_df.to_csv(results_dir / f'{name}_holdings_long_only.csv', index = False)

    predict_at_dates(model, test_dates).to_csv(
        results_dir / f'{name}_test_predictions.csv', index = False,
    )

    return {
        'returns_ls_raw': ls_raw_test, 'returns_ls_scaled': ls_scaled_test,
        'returns_lo_raw': lo_raw_test, 'returns_lo_scaled': lo_scaled_test,
        'dates_ls': ls_dates_test, 'dates_lo': lo_dates_test,
        'metrics': {
            'long_short_raw': m_ls_raw, 'long_short_scaled': m_ls_scaled,
            'long_only_raw': m_lo_raw, 'long_only_scaled': m_lo_scaled,
        },
    }


xgb_eval = evaluate_and_save(xgb_model, 'xgb')
lgb_eval = evaluate_and_save(lgb_model, 'lgb')

for name, ev in [('XGBoost', xgb_eval), ('LightGBM', lgb_eval)]:
    mls = ev['metrics']['long_short_scaled']
    mlo = ev['metrics']['long_only_scaled']
    print(f'{name} long short (scaled): sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
    print(f'{name} long only  (scaled): sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')



xgb_imp = pd.DataFrame({'feature': feature_cols,'importance': xgb_model.feature_importances_}).sort_values('importance', ascending = False)
lgb_imp = pd.DataFrame({'feature': feature_cols,'importance': lgb_model.feature_importances_}).sort_values('importance', ascending = False)

xgb_imp.to_csv(results_dir / 'xgb_feature_importance.csv', index = False)
lgb_imp.to_csv(results_dir / 'lgb_feature_importance.csv', index = False)

print('top 10 xgboost features')
print(xgb_imp.head(10).to_string(index = False))
print('top 10 lightgbm features')
print(lgb_imp.head(10).to_string(index = False))



def _round_or_none(x, ndigits):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), ndigits)


def _strip_per_year(m):
    """Strip the per_year sub block from a metrics dictionary so the JSON
    summary stays compact. The per year breakdown is saved separately as
    a CSV."""
    if not isinstance(m, dict):
        return m
    return {k: v for k, v in m.items() if k != 'per_year'}


def _strip_metrics_block(metrics):
    return {k: _strip_per_year(v) for k, v in metrics.items()}


summary = {
    'construction': 'mean_split_softmax_cap_6m',
    'target_column': ret_col,
    'n_features': len(feature_cols),
    'feature_cols': feature_cols,
    'split': {
        'train': {'start': str(train_dates[0].date()), 'end': str(train_dates[-1].date()),
                  'n_months': len(train_dates), 'n_obs': int(x_train.shape[0])},
        'val': {'start': str(val_dates[0].date()), 'end': str(val_dates[-1].date()),
                'n_months': len(val_dates)},
        'test': {'start': str(test_dates[0].date()), 'end': str(test_dates[-1].date()),
                 'n_months': len(test_dates)},
        'hpo': {'start': str(hpo_dates[0].date()), 'end': str(hpo_dates[-1].date()),
                'n_months': len(hpo_dates), 'n_obs': int(x_hpo.shape[0])},
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
        'optuna_seed': optuna_seed,
        'n_trials_xgb': n_trials_xgb,
        'n_trials_lgb': n_trials_lgb,
    },
    'xgboost': {
        'best_params': xgb_best,
        'best_val_long_short_sharpe': float(xgb_study.best_value) if xgb_study is not None else float(xgb_best_value),
        'rc_val': float(xgb_rc_val), 'rc_test': float(xgb_rc_test),
        'final_training_time_seconds': float(xgb_train_time),
        'portfolio_metrics': _strip_metrics_block(xgb_eval['metrics']),
    },
    'lightgbm': {
        'best_params': lgb_best,
        'best_val_long_short_sharpe': float(lgb_study.best_value) if lgb_study is not None else float(lgb_best_value),
        'rc_val': float(lgb_rc_val), 'rc_test': float(lgb_rc_test),
        'final_training_time_seconds': float(lgb_train_time),
        'portfolio_metrics': _strip_metrics_block(lgb_eval['metrics']),
    },
}

with open(results_dir / 'tree_summary.json', 'w') as gbms:
    json.dump(summary, gbms, indent = 2, default = float)
print(f'summary saved to {results_dir / "tree_summary.json"}')


rows = []
for name, ev, rc in [('xgboost', xgb_eval, xgb_rc_test), ('lightgbm', lgb_eval, lgb_rc_test)]:
    for portfolio, scaling, key in [
        ('long_short', 'unscaled', 'long_short_raw'),
        ('long_short', 'scaled', 'long_short_scaled'),
        ('long_only', 'unscaled', 'long_only_raw'),
        ('long_only', 'scaled', 'long_only_scaled'),
    ]:
        m = ev['metrics'][key]
        rows.append({
            'model': name, 'portfolio': portfolio,
            'scaling': scaling, 'rc_test': round(rc, 4),
            'sharpe': _round_or_none(m['sharpe'], 4),
            'se': _round_or_none(m['se_sharpe'], 4),
            'ann_ret': _round_or_none(m['ann_ret'] * 100, 2),
            'ann_vol': _round_or_none(m['ann_vol'] * 100, 2),
            'cagr': _round_or_none(m['cagr'] * 100, 2),
            'cum_return': _round_or_none(m['cum_return'] * 100, 2),
            'max_dd': _round_or_none(m['max_dd'] * 100, 2),
            'n_obs': m['n_obs'],
        })
summary_table = pd.DataFrame(rows)
print('Tree Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index = False))
summary_table.to_csv(results_dir / 'tree_summary.csv', index = False)
print('summary csv saved')

# per year breakdown across both models. one row per (model, portfolio,
# scaling, year).

per_year_rows = []

def _flush_per_year(model, portfolio, scaling, metrics):
    py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
    for year in sorted(py.keys()):
        ym = py[year]
        per_year_rows.append({
            'model': model, 'portfolio': portfolio,
            'scaling': scaling, 'year': int(year),
            'ann_ret': round(float(ym['ann_ret']) * 100, 4),
            'ann_vol': round(float(ym['ann_vol']) * 100, 4),
            'sharpe': (round(float(ym['sharpe']), 4)
                      if not (isinstance(ym['sharpe'], float) and np.isnan(ym['sharpe']))
                      else None),
            'max_dd': round(float(ym['max_dd']) * 100, 4),
            'cum_return': round(float(ym['cum_return']) * 100, 4),
            'n_obs': int(ym['n_obs'])
        })

for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
    _flush_per_year(name, 'long_short', 'unscaled', ev['metrics']['long_short_raw'])
    _flush_per_year(name, 'long_short', 'scaled', ev['metrics']['long_short_scaled'])
    _flush_per_year(name, 'long_only', 'unscaled', ev['metrics']['long_only_raw'])
    _flush_per_year(name, 'long_only', 'scaled', ev['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'tree_per_year_metrics.csv', index = False)
print(f'per year metrics saved, {len(per_year_df)} rows')


xgb_color = 'steelblue'
lgb_color = 'darkorange'
xlabel_periods = f'Rebalance periods from start of test window ({rebalance_freq} months each)'

# figure 1, volatility targeted cumulative wealth on the scaled series
fig, axes = plt.subplots(1, 2, figsize = (12, 4))
ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_scaled']), label = 'XGBoost', color = xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_scaled']), label = 'LightGBM', color = lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Volatility Targeted')
ax.legend(frameon = False)
ax.grid(alpha = 0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_scaled']), label = 'XGBoost', color = xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_scaled']), label = 'LightGBM', color = lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Volatility Targeted')
ax.legend(frameon = False)
ax.grid(alpha = 0.3)
fig.tight_layout()
plt.show()



# figure 2, unscaled cumulative wealth
fig, axes = plt.subplots(1, 2, figsize = (12, 4))
ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_raw']), label = 'XGBoost', color = xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_raw']), label = 'LightGBM', color = lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Unscaled')
ax.legend(frameon = False)
ax.grid(alpha = 0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_raw']), label = 'XGBoost', color = xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_raw']), label = 'LightGBM', color = lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Unscaled')
ax.legend(frameon = False)
ax.grid(alpha = 0.3)
fig.tight_layout()
plt.show()



# figure 3, scaled and unscaled overlaid, solid is scaled, dashed is unscaled
fig, axes = plt.subplots(1, 2, figsize = (12, 4))
ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_scaled']), label = 'XGBoost, Scaled', color = xgb_color)
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_raw']), label = 'XGBoost, Unscaled', color = xgb_color, linestyle = '--')
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_scaled']), label = 'LightGBM, Scaled', color = lgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_raw']), label = 'LightGBM, Unscaled', color = lgb_color, linestyle = '--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Scaled and Unscaled')
ax.legend(frameon = False, fontsize = 9, loc = 'upper left')
ax.grid(alpha = 0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_scaled']), label = 'XGBoost, Scaled', color = xgb_color)
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_raw']), label = 'XGBoost, Unscaled', color = xgb_color, linestyle = '--')
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_scaled']), label = 'LightGBM, Scaled', color = lgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_raw']), label = 'LightGBM, Unscaled', color = lgb_color, linestyle = '--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Scaled and Unscaled')
ax.legend(frameon = False, fontsize = 9, loc = 'upper left')
ax.grid(alpha = 0.3)
fig.tight_layout()
plt.show()



# figure 4, tree specific diagnostics: optuna search progress and xgboost
# feature importance ranking
fig, axes = plt.subplots(1, 3, figsize = (18, 4))
if xgb_study is not None:
    xgb_vals = [t.value for t in xgb_study.trials if t.value is not None]
    axes[0].plot(np.maximum.accumulate(xgb_vals), color = xgb_color)
    axes[0].scatter(range(len(xgb_vals)), xgb_vals, alpha = 0.3, s = 15, color = xgb_color)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Validation LS Sharpe')
    axes[0].set_title('XGBoost Optuna Search')
    axes[0].grid(alpha = 0.3)
else:
    axes[0].text(0.5, 0.5, 'XGBoost study not in memory', ha = 'center', va = 'center')
    axes[0].set_title('XGBoost Optuna Search')

if lgb_study is not None:
    lgb_vals = [t.value for t in lgb_study.trials if t.value is not None]
    axes[1].plot(np.maximum.accumulate(lgb_vals), color = lgb_color)
    axes[1].scatter(range(len(lgb_vals)), lgb_vals, alpha = 0.3, s = 15, color = lgb_color)
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Validation LS Sharpe')
    axes[1].set_title('LightGBM Optuna Search')
    axes[1].grid(alpha = 0.3)
else:
    axes[1].text(0.5, 0.5, 'LightGBM study not in memory', ha = 'center', va = 'center')
    axes[1].set_title('LightGBM Optuna Search')

top_xgb_imp = xgb_imp.head(15)
axes[2].barh(range(len(top_xgb_imp)), top_xgb_imp['importance'][::-1], color = xgb_color)
axes[2].set_yticks(range(len(top_xgb_imp)))
axes[2].set_yticklabels(top_xgb_imp['feature'][::-1], fontsize = 9)
axes[2].set_xlabel('Importance, Gain')
axes[2].set_title('Top 15 XGBoost Features')
axes[2].grid(axis = 'x', alpha = 0.3)
fig.tight_layout()
plt.show()