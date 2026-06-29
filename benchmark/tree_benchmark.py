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

print(f'cuda available: {cuda_available}')
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
        print(f'{label}: gpu probe failed, {type(exc).__name__}, {exc}')
        return False


xgb_use_cuda = False
lgb_use_gpu = False
if cuda_available:
    xgb_use_cuda = _probe(
        lambda x, y: xgb.XGBRegressor(
            n_estimators=5, tree_method='hist', device='cuda', verbosity=0,
        ).fit(x, y), 'xgboost',
    )
    lgb_use_gpu = _probe(
        lambda x, y: lgb.LGBMRegressor(
            n_estimators=5, device='gpu', verbose=-1,
        ).fit(x, y), 'lightgbm',
    )

xgb_device_params = {'tree_method': 'hist', 'device': 'cuda'} if xgb_use_cuda else {'tree_method': 'hist'}
lgb_device_params = {'device': 'gpu'} if lgb_use_gpu else {}


# configuration

data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/tree_benchmark')
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

n_trials_xgb = 50
n_trials_lgb = 50
optuna_seed = 42
n_hpo_months = 36

periods_per_year = 12.0 / rebalance_freq
n_vol_periods = max(1, vol_lookback_months // rebalance_freq)


# feature schema

schema = pq.read_schema(data_path)

non_feature = {
    'id', 'gvkey', 'iid', 'isin', 'cusip', 'permno', 'permco',
    'eom', 'date', 'excntry', 'curcd', 'size_grp',
    ret_col_1m,
    'sic', 'naics', 'gics', 'ff49',
    'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
    'obs_main', 'exch_main', 'primary_sec', 'common', 'bidask',
    'source_crsp',
    'adjfct', 'fx', 'ret_lag_dif',
    'ret', 'ret_exc', 'ret_local',
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
print(f'countries: {df["excntry"].nunique()}')


# six month cumulative forward target

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
print(f'cumulative six month target constructed')
print(f'retained rows with valid six month forward block: {retained:,} of {len(df):,}')
print(f'retention rate: {100.0 * retained / len(df):.2f}%')

del shifted
gc.collect()


# per month preprocessing: rank normalise each characteristic to the unit
# interval, centre at zero, impute missing to zero (cross sectional median
# after centering). this follows the standard kelly, malamud, zhou, pedersen
# normalisation used by the dppt and all other benchmarks.

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

    all_months[eom] = {'ids': ids, 'r': r, 'x': x}

sorted_dates = sorted(all_months.keys())
print(f'processed: {len(sorted_dates)} months')
print(f'avg firms per month: {np.mean([len(m["ids"]) for m in all_months.values()]):.0f}')


# train, validation, and test splits

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


# portfolio metric helpers
# these mirror the conventions in eval_dual_path.py and fama_french_benchmark.py:
# arithmetic mean for annualised return and sharpe, unbiased std (ddof=0)
# for vol, memmel se for the sharpe estimator. all period returns at six
# month frequency so periods_per_year is 2.

def portfolio_metrics(rets, ppy, dates=None):
    """Annualised metrics for a period return series.

    ppy is periods per year, namely 12 divided by rebalance_freq. The standard
    deviation uses ddof=0, matching eval_dual_path.py. When dates are
    supplied a per_year block is appended, keyed by calendar year. Years with
    fewer than two observations are skipped to avoid degenerate vol estimates."""
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
        'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'se_sharpe': se,
        'max_dd': max_dd, 'cum_return': cum_return,
        'n_obs': n,
    }

    if dates is not None:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        per_year = {}
        for y in sorted(set(years.tolist())):
            mask = years == y
            sub = rets[mask]
            if len(sub) < 2:
                continue
            y_ret = float(sub.mean() * ppy)
            y_vol = float(sub.std() * np.sqrt(ppy))
            y_sharpe = y_ret / max(y_vol, 1e-8)
            ycw = np.cumprod(1.0 + sub)
            ypk = np.maximum.accumulate(ycw)
            y_dd = float(((ypk - ycw) / ypk).max())
            per_year[int(y)] = {
                'ann_ret': y_ret, 'ann_vol': y_vol,
                'sharpe': y_sharpe, 'max_dd': y_dd,
                'cum_return': float(ycw[-1] - 1.0), 'n_obs': int(len(sub)),
            }
        out['per_year'] = per_year

    return out


def _capped_softmax_weights(scores, max_weight, max_iter=20):
    """Iterative capped softmax. Mirrors eval_dual_path._capped_softmax_weights.
    Excess weight above max_weight is redistributed to uncapped positions.
    Falls back to uniform when the cap is mechanically infeasible."""
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
    """Redistribute weight from firms with missing forward returns to those
    with valid forward returns, so the portfolio remains fully invested."""
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
    """One sided turnover as the fraction of the current holding set that
    is new plus the fraction of the previous set that has exited."""
    prev = set(prev_ids.tolist()) if prev_ids is not None else set()
    curr = set(curr_ids.tolist())
    if not curr:
        return 0.0
    new_in = len(curr - prev)
    exited = len(prev - curr)
    return (new_in + exited) / max(len(curr), 1)


def apply_period_vol_overlay(period_rets, n_vol_pds, ppy, max_lev):
    """Volatility overlay on period returns. From period n_vol_pds onward,
    the trailing window of n_vol_pds period returns is used to estimate
    annualised vol, and the leverage factor clips to [1/max_lev, max_lev].
    The estimate is computed on gross unscaled returns, matching
    eval_dual_path.py, so that TC does not suppress the vol signal."""
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


def predict_at_dates(model, month_dates):
    """Per firm predictions across the given dates. Returns a dataframe
    with columns eom, id, prediction, realised_return."""
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
    """Mean split simulation mirroring eval_dual_path.py.

    At each rebalance (every rebalance_freq months):
      Long short: firms above the mean predicted return are long, at or below
        are short. Within each leg, capped softmax weights with max_position_weight
        per position.
      Long only: all firms in the cross section, capped softmax on raw scores.
    Weights are renormalised over firms with valid realised returns. Transaction
    cost is tc_bps basis points per unit of one sided firm id turnover, summed
    over both legs for long short and the long leg only for long only."""
    ls_period_rets, ls_period_dates = [], []
    ls_tc_history = []
    lo_period_rets, lo_period_dates = [], []
    lo_tc_history = []

    prev_long_ids = None
    prev_short_ids = None
    prev_lo_ids = None

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

        long_ret = (
            float(np.sum(long_w[valid[long_idx]] * r[long_idx][valid[long_idx]]))
            if long_idx.size else 0.0
        )
        short_ret = (
            float(np.sum(short_w[valid[short_idx]] * r[short_idx][valid[short_idx]]))
            if short_idx.size else 0.0
        )
        ls_ret = long_ret - short_ret

        lt = _firm_id_turnover(prev_long_ids, long_firm_ids)
        st = _firm_id_turnover(prev_short_ids, short_firm_ids)
        ls_flat_tc = (lt + st) * tc_bps / 10000.0

        ls_period_rets.append(ls_ret)
        ls_period_dates.append(eom)
        ls_tc_history.append(ls_flat_tc)
        prev_long_ids = long_firm_ids
        prev_short_ids = short_firm_ids

        lo_w = _capped_softmax_weights(pred[valid_pred], max_position_weight)
        lo_w_full = np.zeros(n_firms, dtype=np.float64)
        lo_w_full[valid_pred] = lo_w
        lo_w_full = _renorm_over_valid(lo_w_full, valid)
        lo_ret = float(np.sum(lo_w_full[valid] * r[valid]))

        lo_firm_ids = ids[valid_pred]
        lo_turn = _firm_id_turnover(prev_lo_ids, lo_firm_ids)
        lo_flat_tc = lo_turn * tc_bps / 10000.0

        lo_period_rets.append(lo_ret)
        lo_period_dates.append(eom)
        lo_tc_history.append(lo_flat_tc)
        prev_lo_ids = lo_firm_ids

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


def apply_overlay_and_costs(leg_gross_rets, leg_tc, n_vol_pds, ppy, max_lev):
    """Combine gross leg returns with the volatility overlay and transaction costs.

    The overlay is computed on gross returns before TC so the leverage signal
    is not dampened by turnover costs. TC is then scaled by the same leverage
    factor, so the scaled net return is leverage * (gross - tc).

    Returns a tuple (scaled_net, unscaled_net, leverage_path)."""
    leg_gross_rets = np.asarray(leg_gross_rets, dtype=np.float64)
    leg_tc = np.asarray(leg_tc, dtype=np.float64)
    leverage_path = apply_period_vol_overlay(
        leg_gross_rets, n_vol_pds, ppy, max_lev,
    )
    unscaled_net = leg_gross_rets - leg_tc
    scaled_net = leverage_path * leg_gross_rets - leverage_path * leg_tc
    return scaled_net, unscaled_net, leverage_path


def _build_period_rows(model_name, portfolio, scaling, rets, dates):
    """Per rebalance period metrics row builder. Mirrors fama_french_benchmark
    _build_monthly_rows at period frequency. Computes cumulative wealth,
    drawdown, and a trailing four period rolling Sharpe (two years at six
    month rebalance frequency). The rolling window requires at least four
    observations; earlier periods carry None for the rolling Sharpe."""
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


# hyperparameter search

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
    print(f'xgboost best params loaded from {xgb_best_params_path.name}')
    print(f'xgboost best val ls sharpe: {xgb_best_value:.4f}')
else:
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
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
            del model     #type: ignore
            _gpu_cleanup()

    xgb_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )
    t0 = time.time()
    xgb_study.optimize(xgb_objective, n_trials=n_trials_xgb, show_progress_bar=True)
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
            'n_trials_completed': sum(
                1 for t in xgb_study.trials if t.state.name == 'COMPLETE'
            ),
            'hpo_time_seconds': float(xgb_hpo_time),
        }, fh, indent=2, default=float)

    xgb_study.trials_dataframe().to_csv(xgb_trials_path, index=False)
    with open(xgb_study_path, 'wb') as fh:
        pickle.dump(xgb_study, fh)

    print(f'xgboost best val ls sharpe: {xgb_best_value:.4f}')
    print(f'xgboost best params: {xgb_best}')
    print(f'xgboost hpo time: {xgb_hpo_time:.1f} s')


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
    print(f'lightgbm best params loaded from {lgb_best_params_path.name}')
    print(f'lightgbm best val ls sharpe: {lgb_best_value:.4f}')
else:
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
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
            del model                                  #type: ignore
            _gpu_cleanup()

    lgb_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )
    t0 = time.time()
    lgb_study.optimize(lgb_objective, n_trials=n_trials_lgb, show_progress_bar=True)
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
            'n_trials_completed': sum(
                1 for t in lgb_study.trials if t.state.name == 'COMPLETE'
            ),
            'hpo_time_seconds': float(lgb_hpo_time),
        }, fh, indent=2, default=float)

    lgb_study.trials_dataframe().to_csv(lgb_trials_path, index=False)
    with open(lgb_study_path, 'wb') as fh:
        pickle.dump(lgb_study, fh)

    print(f'lightgbm best val ls sharpe: {lgb_best_value:.4f}')
    print(f'lightgbm best params: {lgb_best}')
    print(f'lightgbm hpo time: {lgb_hpo_time:.1f} s')


# final training on train and validation combined

_gpu_cleanup()


class XGBPredictor:
    """Thin wrapper exposing a sklearn style predict, save_model, and
    feature_importances_ on top of a native xgboost booster."""
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
        score = self.booster.get_score(importance_type='gain')
        imp = np.zeros(len(feature_cols), dtype=np.float64)
        for k, v in score.items():
            try:
                idx = int(k.lstrip('f'))
                if 0 <= idx < len(imp):
                    imp[idx] = v
            except ValueError:
                continue
        return imp


class ChunkIter(xgb.DataIter):
    """Feed the training matrix to xgboost in fixed size chunks so that
    peak memory is bounded by the chunk size rather than the full matrix."""
    def __init__(self, x, y, chunk_size, cache_prefix):
        self.x = x
        self.y = y
        self.chunk_size = chunk_size
        self.n_chunks = (len(x) + chunk_size - 1) // chunk_size
        self.current = 0
        super().__init__(cache_prefix=cache_prefix)

    def next(self, input_data):                     #type: ignore
        if self.current >= self.n_chunks:
            return 0
        start = self.current * self.chunk_size
        end = min(start + self.chunk_size, len(self.x))
        input_data(data=self.x[start:end], label=self.y[start:end])
        self.current += 1
        return 1

    def reset(self):
        self.current = 0


xgb_final_params = dict(xgb_best)
xgb_final_params.update({
    'tree_method': 'hist',
    'device': 'cpu',
    'max_bin': 64,
    'objective': 'reg:squarederror',
    'seed': optuna_seed,
    'nthread': -1,
})
n_estimators_xgb = int(xgb_final_params.pop('n_estimators', 100))

cache_dir = results_dir / 'xgb_cache'
cache_dir.mkdir(parents=True, exist_ok=True)

print('building xgboost external memory training matrix')
it = ChunkIter(
    x=x_trainval,
    y=y_trainval,
    chunk_size=200000,
    cache_prefix=str(cache_dir / 'iter'),
)
dtrain = xgb.ExtMemQuantileDMatrix(it, max_bin=64)
print(f'matrix built: {dtrain.num_row():,} rows, {dtrain.num_col()} columns')

del it
gc.collect()

t0 = time.time()
xgb_booster = xgb.train(
    params=xgb_final_params,
    dtrain=dtrain,
    num_boost_round=n_estimators_xgb,
    verbose_eval=False,
)
xgb_train_time = time.time() - t0
xgb_model = XGBPredictor(xgb_booster)

del dtrain
gc.collect()
print(f'xgboost trained in {xgb_train_time:.1f} s')


_gpu_cleanup()

lgb_final_params = {**lgb_device_params, **lgb_best}
lgb_final_params['max_bin'] = 128

lgb_model = lgb.LGBMRegressor(
    **lgb_final_params,
    random_state=optuna_seed,
    n_jobs=-1,
    verbose=-1,
)
t0 = time.time()
lgb_model.fit(x_trainval, y_trainval)
lgb_train_time = time.time() - t0
print(f'lightgbm trained in {lgb_train_time:.1f} s')


_gpu_cleanup()

xgb_model.save_model(str(results_dir / 'xgb_model.json'))
lgb_model.booster_.save_model(str(results_dir / 'lgb_model.txt'))
print('models saved')


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
        c, _ = spearmanr(pred[valid], m['r'][valid])
        if not np.isnan(c):                                                #type: ignore
            corrs.append(float(c))                                        #type: ignore
    return float(np.mean(corrs)) if corrs else 0.0


xgb_rc_val = rank_correlation_oos(xgb_model, val_dates)
xgb_rc_test = rank_correlation_oos(xgb_model, test_dates)
lgb_rc_val = rank_correlation_oos(lgb_model, val_dates)
lgb_rc_test = rank_correlation_oos(lgb_model, test_dates)

print(f'xgboost rank corr: val = {xgb_rc_val:.4f}, test = {xgb_rc_test:.4f}')
print(f'lightgbm rank corr: val = {lgb_rc_val:.4f}, test = {lgb_rc_test:.4f}')


# test set evaluation. the simulation runs on all sorted_dates so the
# volatility overlay accumulates full warm up history before the test
# window begins. the series is then sliced to the test window before
# computing metrics, which matches the approach in fama_french_benchmark.py
# and eval_dual_path.py.

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

    ls_unscaled_test = ls_unscaled_full[ls_mask]
    ls_scaled_test = ls_scaled_full[ls_mask]
    lo_unscaled_test = lo_unscaled_full[lo_mask]
    lo_scaled_test = lo_scaled_full[lo_mask]

    ls_dates_test = [d for d, m in zip(ls['dates'], ls_mask) if m]
    lo_dates_test = [d for d, m in zip(lo['dates'], lo_mask) if m]

    ls_ret_df = pd.DataFrame({
        'eom': ls_dates_test,
        'return_unscaled': ls_unscaled_test,
        'return_scaled': ls_scaled_test,
        'leverage': ls_lev[ls_mask],
    })
    lo_ret_df = pd.DataFrame({
        'eom': lo_dates_test,
        'return_unscaled': lo_unscaled_test,
        'return_scaled': lo_scaled_test,
        'leverage': lo_lev[lo_mask],
    })

    ls_hold_df = (
        ls['holdings_df'][ls['holdings_df']['eom'].isin(test_set)]
        .copy().reset_index(drop=True)
    )
    lo_hold_df = (
        lo['holdings_df'][lo['holdings_df']['eom'].isin(test_set)]
        .copy().reset_index(drop=True)
    )

    m_ls_unscaled = portfolio_metrics(ls_unscaled_test, periods_per_year, dates=ls_dates_test)
    m_ls_scaled = portfolio_metrics(ls_scaled_test, periods_per_year, dates=ls_dates_test)
    m_lo_unscaled = portfolio_metrics(lo_unscaled_test, periods_per_year, dates=lo_dates_test)
    m_lo_scaled = portfolio_metrics(lo_scaled_test, periods_per_year, dates=lo_dates_test)

    ls_ret_df.to_csv(results_dir / f'{name}_returns_long_short.csv', index=False)
    lo_ret_df.to_csv(results_dir / f'{name}_returns_long_only.csv', index=False)
    ls_hold_df.to_csv(results_dir / f'{name}_holdings_long_short.csv', index=False)
    lo_hold_df.to_csv(results_dir / f'{name}_holdings_long_only.csv', index=False)

    predict_at_dates(model, test_dates).to_csv(
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


xgb_eval = evaluate_and_save(xgb_model, 'xgb')
lgb_eval = evaluate_and_save(lgb_model, 'lgb')

for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
    mls = ev['metrics']['long_short_scaled']
    mlo = ev['metrics']['long_only_scaled']
    print(
        f'{name} long short scaled: sharpe = {mls["sharpe"]:.4f}, '
        f'ann_ret = {mls["ann_ret"] * 100:.2f}%, '
        f'ann_vol = {mls["ann_vol"] * 100:.2f}%'
    )
    print(
        f'{name} long only scaled: sharpe = {mlo["sharpe"]:.4f}, '
        f'ann_ret = {mlo["ann_ret"] * 100:.2f}%, '
        f'ann_vol = {mlo["ann_vol"] * 100:.2f}%'
    )


xgb_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_,
}).sort_values('importance', ascending=False)
lgb_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_,
}).sort_values('importance', ascending=False)

xgb_imp.to_csv(results_dir / 'xgb_feature_importance.csv', index=False)
lgb_imp.to_csv(results_dir / 'lgb_feature_importance.csv', index=False)

print('top 10 xgboost features')
print(xgb_imp.head(10).to_string(index=False))
print('top 10 lightgbm features')
print(lgb_imp.head(10).to_string(index=False))


# summary json and csv

def _round_or_none(x, ndigits):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return round(float(x), ndigits)


def _strip_per_year(m):
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
        'hpo': {
            'start': str(hpo_dates[0].date()), 'end': str(hpo_dates[-1].date()),
            'n_months': len(hpo_dates), 'n_obs': int(x_hpo.shape[0]),
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
        'optuna_seed': optuna_seed,
        'n_trials_xgb': n_trials_xgb,
        'n_trials_lgb': n_trials_lgb,
    },
    'xgboost': {
        'best_params': xgb_best,
        'best_val_long_short_sharpe': (
            float(xgb_study.best_value) if xgb_study is not None else float(xgb_best_value)
        ),
        'rc_val': float(xgb_rc_val),
        'rc_test': float(xgb_rc_test),
        'final_training_time_seconds': float(xgb_train_time),
        'portfolio_metrics': _strip_metrics_block(xgb_eval['metrics']),
    },
    'lightgbm': {
        'best_params': lgb_best,
        'best_val_long_short_sharpe': (
            float(lgb_study.best_value) if lgb_study is not None else float(lgb_best_value)
        ),
        'rc_val': float(lgb_rc_val),
        'rc_test': float(lgb_rc_test),
        'final_training_time_seconds': float(lgb_train_time),
        'portfolio_metrics': _strip_metrics_block(lgb_eval['metrics']),
    },
}

with open(results_dir / 'tree_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2, default=float)
print(f'summary json saved')


# headline summary csv. columns match fama_french_benchmark.py: model,
# portfolio, scaling, rc_test (tree specific), sharpe, se, ann_ret, ann_vol,
# cum_return, max_dd, n_obs.

rows = []
for name, ev, rc in [('xgboost', xgb_eval, xgb_rc_test), ('lightgbm', lgb_eval, lgb_rc_test)]:
    for portfolio, scaling, key in [
        ('long_short', 'unscaled', 'long_short_unscaled'),
        ('long_short', 'scaled', 'long_short_scaled'),
        ('long_only', 'unscaled', 'long_only_unscaled'),
        ('long_only', 'scaled', 'long_only_scaled'),
    ]:
        m = ev['metrics'][key]
        rows.append({
            'model': name,
            'portfolio': portfolio,
            'scaling': scaling,
            'rc_test': round(rc, 4),
            'sharpe': _round_or_none(m['sharpe'], 4),
            'se': _round_or_none(m['se_sharpe'], 4),
            'ann_ret': _round_or_none(m['ann_ret'] * 100, 2),
            'ann_vol': _round_or_none(m['ann_vol'] * 100, 2),
            'cum_return': _round_or_none(m['cum_return'] * 100, 2),
            'max_dd': _round_or_none(m['max_dd'] * 100, 2),
            'n_obs': m['n_obs'],
        })

summary_table = pd.DataFrame(rows)
print('\nTree Benchmark, EM Universe, mean split capped softmax, 6m rebalance')
print(summary_table.to_string(index=False))
summary_table.to_csv(results_dir / 'tree_summary.csv', index=False)
print('summary csv saved')


# per year breakdown csv. mirrors fama_french_benchmark.py structure.

per_year_rows = []


def _flush_per_year(model, portfolio, scaling, metrics):
    py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
    for year in sorted(py.keys()):
        ym = py[year]
        per_year_rows.append({
            'model': model,
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


for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
    _flush_per_year(name, 'long_short', 'unscaled', ev['metrics']['long_short_unscaled'])
    _flush_per_year(name, 'long_short', 'scaled', ev['metrics']['long_short_scaled'])
    _flush_per_year(name, 'long_only', 'unscaled', ev['metrics']['long_only_unscaled'])
    _flush_per_year(name, 'long_only', 'scaled', ev['metrics']['long_only_scaled'])

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'tree_per_year_metrics.csv', index=False)
print(f'per year metrics saved, {len(per_year_df)} rows')


# per period metrics csv. mirrors fama_french_benchmark per month metrics csv
# at the six month rebalance frequency of this benchmark. each row is one
# rebalance period for one (model, portfolio, scaling) combination, carrying
# the period return, cumulative wealth, drawdown, and a trailing four period
# rolling sharpe.

period_rows = []

for name, ev in [('xgboost', xgb_eval), ('lightgbm', lgb_eval)]:
    period_rows.extend(_build_period_rows(
        name, 'long_short', 'unscaled',
        ev['returns_ls_unscaled'], ev['dates_ls'],
    ))
    period_rows.extend(_build_period_rows(
        name, 'long_short', 'scaled',
        ev['returns_ls_scaled'], ev['dates_ls'],
    ))
    period_rows.extend(_build_period_rows(
        name, 'long_only', 'unscaled',
        ev['returns_lo_unscaled'], ev['dates_lo'],
    ))
    period_rows.extend(_build_period_rows(
        name, 'long_only', 'scaled',
        ev['returns_lo_scaled'], ev['dates_lo'],
    ))

per_period_df = pd.DataFrame(period_rows)
per_period_df.to_csv(results_dir / 'tree_per_period_metrics.csv', index=False)
print(f'per period metrics saved, {len(per_period_df)} rows')


# plots

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

xgb_color = 'steelblue'
lgb_color = 'darkorange'
xlabel_periods = f'Rebalance periods from start of test window ({rebalance_freq} months each)'

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_scaled']), label='XGBoost', color=xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_scaled']), label='LightGBM', color=lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Volatility Targeted')
ax.legend(frameon=False)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_scaled']), label='XGBoost', color=xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_scaled']), label='LightGBM', color=lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Volatility Targeted')
ax.legend(frameon=False)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / 'tree_cumulative_scaled.pdf')
fig.savefig(results_dir / 'tree_cumulative_scaled.png')
plt.show()
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_unscaled']), label='XGBoost', color=xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_unscaled']), label='LightGBM', color=lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Unscaled')
ax.legend(frameon=False)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_unscaled']), label='XGBoost', color=xgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_unscaled']), label='LightGBM', color=lgb_color)
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Unscaled')
ax.legend(frameon=False)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / 'tree_cumulative_unscaled.pdf')
fig.savefig(results_dir / 'tree_cumulative_unscaled.png')
plt.show()
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_scaled']), label='XGBoost, Scaled', color=xgb_color)
ax.plot(np.cumprod(1 + xgb_eval['returns_ls_unscaled']), label='XGBoost, Unscaled', color=xgb_color, linestyle='--')
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_scaled']), label='LightGBM, Scaled', color=lgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_ls_unscaled']), label='LightGBM, Unscaled', color=lgb_color, linestyle='--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Short, Scaled and Unscaled')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_scaled']), label='XGBoost, Scaled', color=xgb_color)
ax.plot(np.cumprod(1 + xgb_eval['returns_lo_unscaled']), label='XGBoost, Unscaled', color=xgb_color, linestyle='--')
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_scaled']), label='LightGBM, Scaled', color=lgb_color)
ax.plot(np.cumprod(1 + lgb_eval['returns_lo_unscaled']), label='LightGBM, Unscaled', color=lgb_color, linestyle='--')
ax.set_xlabel(xlabel_periods)
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Long Only, Scaled and Unscaled')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / 'tree_cumulative_combined.pdf')
fig.savefig(results_dir / 'tree_cumulative_combined.png')
plt.show()
plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(18, 4))

if xgb_study is not None:
    xgb_vals = [t.value for t in xgb_study.trials if t.value is not None]
    axes[0].plot(np.maximum.accumulate(xgb_vals), color=xgb_color)
    axes[0].scatter(range(len(xgb_vals)), xgb_vals, alpha=0.3, s=15, color=xgb_color)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Validation LS Sharpe')
    axes[0].set_title('XGBoost Optuna Search')
    axes[0].grid(alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'XGBoost study not in memory', ha='center', va='center')
    axes[0].set_title('XGBoost Optuna Search')

if lgb_study is not None:
    lgb_vals = [t.value for t in lgb_study.trials if t.value is not None]
    axes[1].plot(np.maximum.accumulate(lgb_vals), color=lgb_color)
    axes[1].scatter(range(len(lgb_vals)), lgb_vals, alpha=0.3, s=15, color=lgb_color)
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Validation LS Sharpe')
    axes[1].set_title('LightGBM Optuna Search')
    axes[1].grid(alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'LightGBM study not in memory', ha='center', va='center')
    axes[1].set_title('LightGBM Optuna Search')

top_xgb_imp = xgb_imp.head(15)
axes[2].barh(range(len(top_xgb_imp)), top_xgb_imp['importance'][::-1], color=xgb_color)
axes[2].set_yticks(range(len(top_xgb_imp)))
axes[2].set_yticklabels(top_xgb_imp['feature'][::-1], fontsize=9)
axes[2].set_xlabel('Importance, Gain')
axes[2].set_title('Top 15 XGBoost Features')
axes[2].grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / 'tree_diagnostics.pdf')
fig.savefig(results_dir / 'tree_diagnostics.png')
plt.show()
plt.close(fig)

print('plots saved')
