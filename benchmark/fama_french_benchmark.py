
## Fama-French Five-Factor Benchmark

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


# configuration
data_path = Path('data/Global Factor_EM.parquet')
results_dir = Path('results/benchmark/ff_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

# jkp column names for the five fama-french factor proxies
char_map = {
    'value': 'be_me',
    'profitability': 'ope_be',
    'investment': 'at_gr1',
    'momentum': 'ret_12_1',
    'size': 'me',
}
fm_chars = list(char_map.values())

ret_col = 'ret_exc_lead1m'
rebalance_freq = 3
tc_bps = 25
min_stocks = 30
ret_clip_low = -1.0
ret_clip_high = 1.0

target_vol = 0.10
vol_lookback_months = 12
max_leverage_ls = 3.0
max_leverage_lo = 3.0

id_cols = ['id', 'eom', 'excntry', ret_col, 'me']

test_start = pd.Timestamp('2021-01-01')

## Load and process

schema = pq.read_schema(data_path)
all_col_names = schema.names

factor_cols = list(char_map.values())
fm_available = [c for c in fm_chars if c in all_col_names]
all_chars = list(dict.fromkeys(factor_cols + fm_available))
needed = list(dict.fromkeys([c for c in id_cols + all_chars if c in all_col_names]))

df = pd.read_parquet(data_path, columns = needed)
df['eom'] = pd.to_datetime(df['eom'])

print(f'rows loaded, {df.shape[0]:,}')
print(f'columns loaded, {df.shape[1]}')
print(f'date range, {df["eom"].min().date()} to {df["eom"].max().date()}')

for col in all_chars:
    if col in df.columns and df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)
if 'me' in df.columns and df['me'].dtype == np.float64:
    df['me'] = df['me'].astype(np.float32)

df[ret_col] = df[ret_col].clip(lower = ret_clip_low, upper = ret_clip_high)


## Build monthly cross-sections

sorted_eoms = sorted(df['eom'].unique())
all_months = {}

for eom in sorted_eoms:
    month = df[df['eom'] == eom].copy()
    if len(month) < min_stocks:
        continue

    entry = {
        'ids': month['id'].values,
        'r': month[ret_col].values.astype(np.float64),
        'me': (month['me'].values.astype(np.float64)
                if 'me' in month.columns else np.ones(len(month))),
    }

    for fname, cname in char_map.items():
        entry[fname] = (month[cname].values.astype(np.float64)
                        if cname in month.columns else np.full(len(month), np.nan))

    fm_vals = {}
    fm_valid = {}
    for cname in fm_available:
        vals = (month[cname].values.astype(np.float64)
                if cname in month.columns else np.full(len(month), np.nan))
        valid = np.isfinite(vals)
        ranked = np.zeros(len(month))
        if valid.sum() > 5:
            ranked[valid] = pd.Series(vals[valid]).rank(pct = True).values - 0.5              #type:ignore
        fm_vals[cname] = ranked
        fm_valid[cname] = valid

    if fm_available:
        entry['fm_x'] = np.column_stack([fm_vals[c] for c in fm_available])
        entry['fm_x_valid'] = np.column_stack([fm_valid[c] for c in fm_available])
    else:
        entry['fm_x'] = np.empty((len(month), 0), dtype = np.float64)
        entry['fm_x_valid'] = np.ones((len(month), 0), dtype = bool)
    all_months[eom] = entry

sorted_dates = sorted(all_months.keys())
print(f'processed months, {len(sorted_dates)}')


def portfolio_metrics(rets, dates = None):
    rets = np.array(rets, dtype = np.float64)
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
    ann_ret = float(rets.mean() * 12.0)
    ann_vol = float(rets.std() * np.sqrt(12.0))
    sharpe = ann_ret / max(ann_vol, 1e-8)
    se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / n))
    cw = np.cumprod(1.0 + rets)
    pk = np.maximum.accumulate(cw)
    max_dd = float(((pk - cw) / pk).max())
    cum_return = float(cw[-1] - 1.0)

    out = {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'se_sharpe': se, 'max_dd': max_dd, 'cum_return': cum_return,
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
            y_ret = float(sub.mean() * 12.0)
            y_vol = float(sub.std() * np.sqrt(12.0))
            y_sharpe = y_ret / max(y_vol, 1e-8)
            ycw = np.cumprod(1.0 + sub)
            ypk = np.maximum.accumulate(ycw)
            y_dd = float(((ypk - ycw) / ypk).max())
            per_year[int(y)] = {
                'ann_ret': y_ret, 'ann_vol': y_vol, 'sharpe': y_sharpe,
                'max_dd': y_dd, 'cum_return': float(ycw[-1] - 1.0),
                'n_obs': int(len(sub)),
            }
        out['per_year'] = per_year

    return out


def apply_vol_target(monthly_rets, rebalance_indices, target_vol, lookback_months, max_leverage):
    scaled = np.array(monthly_rets, dtype = np.float64)
    n = len(monthly_rets)
    n_rb = len(rebalance_indices)
    for i in range(n_rb):
        rb_idx = rebalance_indices[i]
        start_month = max(0, rb_idx - lookback_months)
        trailing = np.array(monthly_rets[start_month:rb_idx], dtype = np.float64)
        if len(trailing) < lookback_months:
            continue
        sigma_ann = float(trailing.std() * np.sqrt(12.0))
        lev = float(np.clip(target_vol / max(sigma_ann, 1e-8), 1.0 / max_leverage, max_leverage))
        next_rb = rebalance_indices[i + 1] if i + 1 < n_rb else n
        scaled[rebalance_indices[i]:next_rb] = (
            np.array(monthly_rets[rebalance_indices[i]:next_rb]) * lev
        )
    return scaled


def filter_to_test_window(rets, dates, start_date):
    rets_arr = np.array(rets, dtype = np.float64)
    dates_arr = pd.DatetimeIndex(dates)
    mask = dates_arr >= start_date
    return rets_arr[mask], list(dates_arr[mask])


def selected_weight_map(ids, selected_ids, weight_vals = None, gross = 1.0):
    selected_ids = set(selected_ids)
    if not selected_ids:
        return {}

    id_arr = np.asarray(ids)
    selected = np.array([sid in selected_ids for sid in id_arr.tolist()])
    weights = np.ones(len(id_arr), dtype = np.float64)
    if weight_vals is not None:
        weights = np.asarray(weight_vals, dtype = np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    selected_weights = weights[selected]
    selected_ids_arr = id_arr[selected]
    weight_sum = selected_weights.sum()
    if weight_sum <= 0:
        selected_weights = np.ones(len(selected_ids_arr), dtype = np.float64)
        weight_sum = selected_weights.sum()

    return {
        sid: float(gross * w / weight_sum)
        for sid, w in zip(selected_ids_arr.tolist(), selected_weights)
    }


def portfolio_turnover(new_weights, old_weights):
    ids = set(new_weights) | set(old_weights)
    return float(sum(abs(new_weights.get(sid, 0.0) - old_weights.get(sid, 0.0)) for sid in ids))


def selected_portfolio_return(ids, rets, selected_ids, weight_vals = None):
    selected_ids = set(selected_ids)
    if not selected_ids:
        return 0.0

    id_arr = np.asarray(ids)
    selected = np.array([sid in selected_ids for sid in id_arr.tolist()])
    finite_rets = np.isfinite(rets)
    selected = selected & finite_rets
    selected_rets = rets[selected]
    selected_rets = selected_rets[np.isfinite(selected_rets)]
    if len(selected_rets) == 0:
        return 0.0

    if weight_vals is None:
        return float(selected_rets.mean())

    selected_weights = np.asarray(weight_vals, dtype = np.float64)[selected]
    selected_weights = np.where(np.isfinite(selected_weights) & (selected_weights > 0), selected_weights, 0.0)
    weight_sum = selected_weights.sum()
    if weight_sum <= 0:
        return float(selected_rets.mean())
    return float((selected_weights / weight_sum * selected_rets).sum())


## Market portfolio (value-weighted, long-only by construction)
market_rets = []
market_dates = []
for eom in sorted_dates:
    m = all_months[eom]
    valid_me = np.isfinite(m['me']) & (m['me'] > 0)
    valid_ret = valid_me & np.isfinite(m['r'])
    if valid_me.sum() < 5 or valid_ret.sum() < 5:
        continue
    market_rets.append(selected_portfolio_return(
        m['ids'], m['r'], set(m['ids'][valid_me].tolist()), weight_vals = m['me'],
    ))
    market_dates.append(eom)

market_rets_full = np.array(market_rets)

# the market portfolio rebalances every month (it is reweighted by market
# cap each month). the vol overlay uses the same trailing window length as
# the factor portfolios so the scaled column is comparable across rows.
market_rb_indices = list(range(len(market_rets_full)))
market_scaled_full = apply_vol_target(
    market_rets_full, market_rb_indices, target_vol, vol_lookback_months, max_leverage_lo,
)

# filter both series to the test window before computing metrics
market_rets, market_dates_test = filter_to_test_window(market_rets_full, market_dates, test_start)
market_scaled, _ = filter_to_test_window(market_scaled_full, market_dates, test_start)

mkt_m = portfolio_metrics(market_rets, dates = market_dates_test)
mkt_m_scaled = portfolio_metrics(market_scaled, dates = market_dates_test)

print(f'market months in test window, {len(market_rets)}')
print(f'market unscaled, sharpe = {mkt_m["sharpe"]:.4f}, ann_ret = {mkt_m["ann_ret"] * 100:.2f}%, ann_vol = {mkt_m["ann_vol"] * 100:.2f}%')
print(f'market scaled, sharpe = {mkt_m_scaled["sharpe"]:.4f}, ann_ret = {mkt_m_scaled["ann_ret"] * 100:.2f}%, ann_vol = {mkt_m_scaled["ann_vol"] * 100:.2f}%')


## Sorted factor portfolios (long-short and long-only)

def sorted_factor_portfolio(factor_name, reverse = False, tail_frac = 0.30):
    if sorted_dates:
        start = pd.Timestamp(sorted_dates[0])
        rset = {
            d for d in sorted_dates
            if (
                (pd.Timestamp(d).year - start.year) * 12
                + (pd.Timestamp(d).month - start.month)
            ) % rebalance_freq == 0
        }
    else:
        rset = set()
    ls_rets, ls_dates, ls_rb_indices = [], [], []
    lo_rets, lo_dates, lo_rb_indices = [], [], []
    li_ids, si_ids = set(), set()
    prev_ls_weights, prev_lo_weights = {}, {}

    for eom in sorted_dates:
        m = all_months[eom]
        ids = m['ids']
        r = m['r']
        char_vals = m.get(factor_name)
        if char_vals is None:
            continue

        ls_tcv = 0.0
        lo_tcv = 0.0

        if eom in rset:
            ls_rb_indices.append(len(ls_rets))
            lo_rb_indices.append(len(lo_rets))
            valid = np.isfinite(char_vals)
            if valid.sum() < 10:
                ls_exit_tcv = portfolio_turnover({}, prev_ls_weights) * tc_bps / 10000.0
                lo_exit_tcv = portfolio_turnover({}, prev_lo_weights) * tc_bps / 10000.0
                prev_ls_weights, prev_lo_weights = {}, {}
                li_ids, si_ids = set(), set()
                ls_rets.append(-ls_exit_tcv)
                ls_dates.append(eom)
                lo_rets.append(-lo_exit_tcv)
                lo_dates.append(eom)
                continue
            vi = ids[valid]
            vc = char_vals[valid]
            nq = max(1, int(len(vi) * tail_frac))
            so = np.argsort(vc)
            if reverse:
                li_ids = set(vi[so[:nq]].tolist())
                si_ids = set(vi[so[::-1][:nq]].tolist())
            else:
                li_ids = set(vi[so[::-1][:nq]].tolist())
                si_ids = set(vi[so[:nq]].tolist())

            long_weights = selected_weight_map(ids, li_ids, weight_vals = m['me'], gross = 1.0)
            short_weights = selected_weight_map(ids, si_ids, weight_vals = m['me'], gross = -1.0)
            ls_weights = {**long_weights, **short_weights}
            lo_weights = long_weights
            ls_tcv = portfolio_turnover(ls_weights, prev_ls_weights) * tc_bps / 10000.0
            lo_tcv = portfolio_turnover(lo_weights, prev_lo_weights) * tc_bps / 10000.0

            prev_ls_weights = ls_weights
            prev_lo_weights = lo_weights

        if not li_ids:
            continue
        lr_mean = selected_portfolio_return(ids, r, li_ids, weight_vals = m['me'])
        sr_mean = selected_portfolio_return(ids, r, si_ids, weight_vals = m['me'])
        ls_rets.append(lr_mean - sr_mean - ls_tcv)
        ls_dates.append(eom)
        lo_rets.append(lr_mean - lo_tcv)
        lo_dates.append(eom)

    return {
        'long_short': {'returns': np.array(ls_rets), 'dates': ls_dates, 'rb_indices': ls_rb_indices},
        'long_only': {'returns': np.array(lo_rets), 'dates': lo_dates, 'rb_indices': lo_rb_indices},
    }

# run the five factor portfolios and report per factor metrics on the test window

factor_defs = [
    ('value', False), ('momentum', False), ('profitability', False),
    ('investment', True), ('size', True),
]

factor_results = {}
rows_for_table = []

for fname, rev in factor_defs:
    tail_frac = 0.50 if fname == 'size' else 0.30
    sim = sorted_factor_portfolio(fname, reverse = rev, tail_frac = tail_frac)
    ls, lo = sim['long_short'], sim['long_only']
    if len(ls['returns']) == 0:
        print(f'factor, {fname}, no data')
        continue
    # vol overlay applied to the full sample so the trailing window estimator
    # has full warm up before the test window starts
    ls_scaled_full = apply_vol_target(ls['returns'], ls['rb_indices'], target_vol, vol_lookback_months, max_leverage_ls)
    lo_scaled_full = apply_vol_target(lo['returns'], lo['rb_indices'], target_vol, vol_lookback_months, max_leverage_lo)

    # filter both raw and scaled to the test window
    ls_rets_test, ls_dates_test = filter_to_test_window(ls['returns'], ls['dates'], test_start)
    lo_rets_test, lo_dates_test = filter_to_test_window(lo['returns'], lo['dates'], test_start)
    ls_scaled_test, _ = filter_to_test_window(ls_scaled_full, ls['dates'], test_start)
    lo_scaled_test, _ = filter_to_test_window(lo_scaled_full, lo['dates'], test_start)

    factor_results[fname] = {
        'returns_ls_unscaled': ls_rets_test, 'returns_ls_scaled': ls_scaled_test,
        'returns_lo_unscaled': lo_rets_test, 'returns_lo_scaled': lo_scaled_test,
        'dates_ls': ls_dates_test, 'dates_lo': lo_dates_test,
        'metrics_ls_unscaled': portfolio_metrics(ls_rets_test, dates = ls_dates_test),
        'metrics_ls_scaled': portfolio_metrics(ls_scaled_test, dates = ls_dates_test),
        'metrics_lo_unscaled': portfolio_metrics(lo_rets_test, dates = lo_dates_test),
        'metrics_lo_scaled': portfolio_metrics(lo_scaled_test, dates = lo_dates_test),
    }
    mls = factor_results[fname]['metrics_ls_scaled']
    mlo = factor_results[fname]['metrics_lo_scaled']
    rows_for_table.append({
        'factor': fname,
        'ls_sharpe': round(mls['sharpe'], 4),
        'ls_ann_ret': round(mls['ann_ret'] * 100, 2),
        'ls_ann_vol': round(mls['ann_vol'] * 100, 2),
        'lo_sharpe': round(mlo['sharpe'], 4),
        'lo_ann_ret': round(mlo['ann_ret'] * 100, 2),
        'lo_ann_vol': round(mlo['ann_vol'] * 100, 2),
    })

factor_table = pd.DataFrame(rows_for_table)
print(f'Sorted Factor Portfolios, test window from {test_start.date()}, vol targeted')
print(factor_table.to_string(index = False))

## Fama-macbeth cross-sectional regression

fm_betas = []
fm_dates_used = []

for eom in sorted_dates:
    m = all_months[eom]
    x = m['fm_x']
    r = m['r']
    valid = np.isfinite(r)
    for j in range(x.shape[1]):
        valid = valid & np.isfinite(x[:, j])
    if valid.sum() < len(fm_available) + 5:
        continue
    x_aug = np.column_stack([np.ones(valid.sum()), x[valid]])
    try:
        beta = np.linalg.lstsq(x_aug, r[valid], rcond = None)[0]
        fm_betas.append(beta)
        fm_dates_used.append(eom)
    except np.linalg.LinAlgError:
        continue

fm_betas = np.array(fm_betas, dtype = np.float64)
if fm_betas.size == 0:
    fm_betas = np.empty((0, 1 + len(fm_available)), dtype = np.float64)
n_months_fm = len(fm_betas)
if n_months_fm > 0:
    fm_mean = fm_betas.mean(axis = 0)
    if n_months_fm > 1:
        fm_se = fm_betas.std(axis = 0, ddof = 1) / np.sqrt(n_months_fm)
        fm_tstat = fm_mean / np.maximum(fm_se, 1e-10)
    else:
        fm_se = np.full(1 + len(fm_available), np.nan)
        fm_tstat = np.full(1 + len(fm_available), np.nan)
else:
    fm_mean = np.full(1 + len(fm_available), np.nan)
    fm_se = np.full(1 + len(fm_available), np.nan)
    fm_tstat = np.full(1 + len(fm_available), np.nan)

coef_names = ['intercept'] + fm_available
fm_results_table = []
for i, name in enumerate(coef_names):
    if n_months_fm > 1 and np.isfinite(fm_tstat[i]):
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(fm_tstat[i]), df = n_months_fm - 1))
    else:
        p_val = np.nan
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
    fm_results_table.append({
        'variable': name,
        'mean_coef': round(float(fm_mean[i]), 5),
        'se': round(float(fm_se[i]), 5),
        't_stat': round(float(fm_tstat[i]), 4),
        'p_value': round(float(p_val), 4),
        'sig': sig,
    })

fm_coef_df = pd.DataFrame(fm_results_table)
print(f'Fama-MacBeth Regression, {n_months_fm} months, {len(fm_available)} characteristics')
print(fm_coef_df.to_string(index = False))


## FM predictive portfolio (long-short and long-only)

min_history = 60
fm_predictions = {}

for t_idx in range(min_history, len(fm_dates_used)):
    beta_avg = fm_betas[:t_idx].mean(axis = 0)
    pred_date = fm_dates_used[t_idx]
    m = all_months[pred_date]
    pred = beta_avg[0] + m['fm_x'] @ beta_avg[1:]
    fully_observed = m['fm_x_valid'].all(axis = 1)
    valid = np.isfinite(pred) & fully_observed
    if valid.sum() < 10:
        continue
    fm_predictions[pred_date] = {
        'w': pred[valid].astype(np.float32),
        'ids': m['ids'][valid],
        'me': m['me'][valid].astype(np.float32),
        'r': m['r'][valid].astype(np.float32),
    }

print(f'FM predictive portfolio months out-of-sample, {len(fm_predictions)}')

# rank correlation computed on the test window only, matching the metric basis
fm_corrs = []
for date_key, p in fm_predictions.items():
    if date_key < test_start:
        continue
    valid_corr = np.isfinite(p['w']) & np.isfinite(p['r'])
    if valid_corr.sum() < 10:
        continue
    c, _ = spearmanr(p['w'][valid_corr], p['r'][valid_corr])
    if not np.isnan(c):                                                #type:ignore
        fm_corrs.append(float(c))                                    #type:ignore
fm_rc = float(np.mean(fm_corrs)) if fm_corrs else 0.0
print(f'FM rank correlation, {fm_rc:.4f}')

keys_fm = sorted(fm_predictions.keys())

if keys_fm:
    start_fm = pd.Timestamp(keys_fm[0])
    rset_fm = {
        d for d in keys_fm
        if (
            (pd.Timestamp(d).year - start_fm.year) * 12
            + (pd.Timestamp(d).month - start_fm.month)
        ) % rebalance_freq == 0
    }
else:
    rset_fm = set()

fm_ls_rets, fm_ls_dates, fm_ls_rb_indices = [], [], []
fm_lo_rets, fm_lo_dates, fm_lo_rb_indices = [], [], []
li_ids, si_ids = set(), set()
prev_ls_weights, prev_lo_weights = {}, {}

for eom in keys_fm:
    p = fm_predictions[eom]
    w = p['w']
    ids = p['ids']
    me = p['me']
    r = p['r']

    ls_tcv = 0.0
    lo_tcv = 0.0

    if eom in rset_fm:
        fm_ls_rb_indices.append(len(fm_ls_rets))
        fm_lo_rb_indices.append(len(fm_lo_rets))
        nq = max(1, int(len(w) * 0.30))
        so = np.argsort(w)
        li_ids = set(ids[so[::-1][:nq]].tolist())
        si_ids = set(ids[so[:nq]].tolist())

        long_weights = selected_weight_map(ids, li_ids, weight_vals = me, gross = 1.0)
        short_weights = selected_weight_map(ids, si_ids, weight_vals = me, gross = -1.0)
        ls_weights = {**long_weights, **short_weights}
        lo_weights = long_weights
        ls_tcv = portfolio_turnover(ls_weights, prev_ls_weights) * tc_bps / 10000.0
        lo_tcv = portfolio_turnover(lo_weights, prev_lo_weights) * tc_bps / 10000.0

        prev_ls_weights = ls_weights
        prev_lo_weights = lo_weights

    if not li_ids:
        continue
    lr_mean = selected_portfolio_return(ids, r, li_ids, weight_vals = me)
    sr_mean = selected_portfolio_return(ids, r, si_ids, weight_vals = me)
    fm_ls_rets.append(lr_mean - sr_mean - ls_tcv)
    fm_ls_dates.append(eom)
    fm_lo_rets.append(lr_mean - lo_tcv)
    fm_lo_dates.append(eom)

fm_ls_rets_full = np.array(fm_ls_rets)
fm_lo_rets_full = np.array(fm_lo_rets)
fm_ls_scaled_full = apply_vol_target(fm_ls_rets_full, fm_ls_rb_indices, target_vol, vol_lookback_months, max_leverage_ls)
fm_lo_scaled_full = apply_vol_target(fm_lo_rets_full, fm_lo_rb_indices, target_vol, vol_lookback_months, max_leverage_lo)

# filter the four series to the test window before computing metrics
fm_ls_rets, fm_ls_dates_test = filter_to_test_window(fm_ls_rets_full, fm_ls_dates, test_start)
fm_lo_rets, fm_lo_dates_test = filter_to_test_window(fm_lo_rets_full, fm_lo_dates, test_start)
fm_ls_scaled, _ = filter_to_test_window(fm_ls_scaled_full, fm_ls_dates, test_start)
fm_lo_scaled, _ = filter_to_test_window(fm_lo_scaled_full, fm_lo_dates, test_start)

fm_ls_unscaled_m = portfolio_metrics(fm_ls_rets, dates = fm_ls_dates_test)
fm_ls_scaled_m = portfolio_metrics(fm_ls_scaled, dates = fm_ls_dates_test)
fm_lo_unscaled_m = portfolio_metrics(fm_lo_rets, dates = fm_lo_dates_test)
fm_lo_scaled_m = portfolio_metrics(fm_lo_scaled, dates = fm_lo_dates_test)

print(f'FM long-short unscaled, sharpe = {fm_ls_unscaled_m["sharpe"]:.4f}')
print(f'FM long-short scaled, sharpe = {fm_ls_scaled_m["sharpe"]:.4f}, ann_ret = {fm_ls_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {fm_ls_scaled_m["ann_vol"] * 100:.2f}%')
print(f'FM long-only unscaled, sharpe = {fm_lo_unscaled_m["sharpe"]:.4f}')
print(f'FM long-only scaled, sharpe = {fm_lo_scaled_m["sharpe"]:.4f}, ann_ret = {fm_lo_scaled_m["ann_ret"] * 100:.2f}%, ann_vol = {fm_lo_scaled_m["ann_vol"] * 100:.2f}%')

## Save Results

for fname, fr in factor_results.items():
    np.save(results_dir / f'{fname}_returns_ls_unscaled.npy', fr['returns_ls_unscaled'])
    np.save(results_dir / f'{fname}_returns_ls_scaled.npy', fr['returns_ls_scaled'])
    np.save(results_dir / f'{fname}_returns_lo_unscaled.npy', fr['returns_lo_unscaled'])
    np.save(results_dir / f'{fname}_returns_lo_scaled.npy', fr['returns_lo_scaled'])

np.save(results_dir / 'market_returns.npy', market_rets)
np.save(results_dir / 'fm_returns_ls_unscaled.npy', fm_ls_rets)
np.save(results_dir / 'fm_returns_ls_scaled.npy', fm_ls_scaled)
np.save(results_dir / 'fm_returns_lo_unscaled.npy', fm_lo_rets)
np.save(results_dir / 'fm_returns_lo_scaled.npy', fm_lo_scaled)

fm_beta_df = pd.DataFrame(
    fm_betas, columns = ['intercept'] + fm_available,
    index = pd.DatetimeIndex(fm_dates_used),
)
fm_beta_df.to_csv(results_dir / 'fm_monthly_betas.csv')


# per year metrics table. one row per (strategy, portfolio, scaling, year).
# this file is the basis for the year by year diagnostic plots.

per_year_rows = []

def _flush_per_year(strategy, portfolio, scaling, metrics):
    py = metrics.get('per_year', {}) if isinstance(metrics, dict) else {}
    for year in sorted(py.keys()):
        ym = py[year]
        per_year_rows.append({
            'strategy': strategy, 'portfolio': portfolio,
            'scaling': scaling, 'year': int(year),
            'ann_ret': round(float(ym['ann_ret']) * 100, 4),
            'ann_vol': round(float(ym['ann_vol']) * 100, 4),
            'sharpe': round(float(ym['sharpe']), 4),
            'max_dd': round(float(ym['max_dd']) * 100, 4),
            'cum_return': round(float(ym['cum_return']) * 100, 4),
            'n_obs': int(ym['n_obs']),
        })

_flush_per_year('market_value_weighted', 'long_only', 'unscaled', mkt_m)
_flush_per_year('market_value_weighted', 'long_only', 'scaled', mkt_m_scaled)

for fname, fr in factor_results.items():
    _flush_per_year(fname, 'long_short', 'unscaled', fr['metrics_ls_unscaled'])
    _flush_per_year(fname, 'long_short', 'scaled', fr['metrics_ls_scaled'])
    _flush_per_year(fname, 'long_only', 'unscaled', fr['metrics_lo_unscaled'])
    _flush_per_year(fname, 'long_only', 'scaled', fr['metrics_lo_scaled'])

_flush_per_year('fm_regression', 'long_short', 'unscaled', fm_ls_unscaled_m)
_flush_per_year('fm_regression', 'long_short', 'scaled', fm_ls_scaled_m)
_flush_per_year('fm_regression', 'long_only', 'unscaled', fm_lo_unscaled_m)
_flush_per_year('fm_regression', 'long_only', 'scaled', fm_lo_scaled_m)

per_year_df = pd.DataFrame(per_year_rows)
per_year_df.to_csv(results_dir / 'ff_per_year_metrics.csv', index = False)
print(f'per year metrics saved, {len(per_year_df)} rows')


def _build_monthly_rows(strategy, portfolio, scaling, rets, dates):
    rets = np.asarray(rets, dtype = np.float64)
    if len(rets) == 0:
        return []
    cum_wealth = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak

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
            'strategy': strategy,
            'portfolio': portfolio,
            'scaling': scaling,
            'eom': pd.Timestamp(eom).strftime('%Y-%m-%d'),
            'return': round(float(rets[i]), 6),
            'cumulative_wealth': round(float(cum_wealth[i]), 6),
            'drawdown': round(float(drawdown[i]), 6),
            'rolling_sharpe_12m': (
                None if np.isnan(rolling_sharpe[i]) else round(float(rolling_sharpe[i]), 4)
            ),
            'rolling_return_12m': (
                None if np.isnan(rolling_ret[i]) else round(float(rolling_ret[i]) * 100, 4)
            ),
        })
    return rows


monthly_rows = []
monthly_rows.extend(_build_monthly_rows(
    'market_value_weighted', 'long_only', 'unscaled', market_rets, market_dates_test,
))
monthly_rows.extend(_build_monthly_rows(
    'market_value_weighted', 'long_only', 'scaled', market_scaled, market_dates_test,
))

for fname, fr in factor_results.items():
    monthly_rows.extend(_build_monthly_rows(
        fname, 'long_short', 'unscaled', fr['returns_ls_unscaled'], fr['dates_ls'],
    ))
    monthly_rows.extend(_build_monthly_rows(
        fname, 'long_short', 'scaled', fr['returns_ls_scaled'], fr['dates_ls'],
    ))
    monthly_rows.extend(_build_monthly_rows(
        fname, 'long_only', 'unscaled', fr['returns_lo_unscaled'], fr['dates_lo'],
    ))
    monthly_rows.extend(_build_monthly_rows(
        fname, 'long_only', 'scaled', fr['returns_lo_scaled'], fr['dates_lo'],
    ))

monthly_rows.extend(_build_monthly_rows(
    'fm_regression', 'long_short', 'unscaled', fm_ls_rets, fm_ls_dates_test,
))
monthly_rows.extend(_build_monthly_rows(
    'fm_regression', 'long_short', 'scaled', fm_ls_scaled, fm_ls_dates_test,
))
monthly_rows.extend(_build_monthly_rows(
    'fm_regression', 'long_only', 'unscaled', fm_lo_rets, fm_lo_dates_test,
))
monthly_rows.extend(_build_monthly_rows(
    'fm_regression', 'long_only', 'scaled', fm_lo_scaled, fm_lo_dates_test,
))

per_month_df = pd.DataFrame(monthly_rows)
per_month_df.to_csv(results_dir / 'ff_per_month_metrics.csv', index = False)
print(f'per month metrics saved, {len(per_month_df)} rows')


def _strip_per_year(m):
    if not isinstance(m, dict):
        return m
    return {k: v for k, v in m.items() if k != 'per_year'}

summary = {
    'universe': 'EM',
    'fm_characteristics': fm_available,
    'test_start': str(test_start.date()),
    'market_long_only_unscaled': _strip_per_year(mkt_m),
    'market_long_only_scaled': _strip_per_year(mkt_m_scaled),
    'factors': {
        f: {
            'long_short_scaled': _strip_per_year(fr['metrics_ls_scaled']),
            'long_short_unscaled': _strip_per_year(fr['metrics_ls_unscaled']),
            'long_only_scaled': _strip_per_year(fr['metrics_lo_scaled']),
            'long_only_unscaled': _strip_per_year(fr['metrics_lo_unscaled']),
        }
        for f, fr in factor_results.items()
    },
    'fm_regression': {
        'n_months': n_months_fm, 'n_chars': len(fm_available),
        'characteristics': fm_available, 'coefficients': fm_results_table,
    },
    'fm_portfolio': {
        'long_short_unscaled': _strip_per_year(fm_ls_unscaled_m),
        'long_short_scaled': _strip_per_year(fm_ls_scaled_m),
        'long_only_unscaled': _strip_per_year(fm_lo_unscaled_m),
        'long_only_scaled': _strip_per_year(fm_lo_scaled_m),
        'rank_corr': fm_rc, 'n_oos_months': len(fm_predictions),
    },
}
with open(results_dir / 'ff_summary.json', 'w') as fh:
    json.dump(summary, fh, indent = 2, default = float)
print(f'summary json saved')


def _row(strategy, portfolio, scaling, m):
    return {
        'strategy': strategy, 'portfolio': portfolio,
        'scaling': scaling, 'sharpe': round(m['sharpe'], 4),
        'se': round(m['se_sharpe'], 4), 'ann_ret': round(m['ann_ret'] * 100, 2),
        'ann_vol': round(m['ann_vol'] * 100, 2), 'cum_return': round(m['cum_return'] * 100, 2),
        'max_dd': round(m['max_dd'] * 100, 2), 'n_obs': m['n_obs'],
    }


summary_rows = []

for scaling, mk in [
    ('unscaled', mkt_m),
    ('scaled', mkt_m_scaled),
]:
    summary_rows.append(_row('market_value_weighted', 'long_only', scaling, mk))

for fname in ['value', 'momentum', 'profitability', 'investment', 'size']:
    if fname not in factor_results:
        continue
    for portfolio, scaling, mkey in [
        ('long_short', 'unscaled', 'metrics_ls_unscaled'),
        ('long_short', 'scaled', 'metrics_ls_scaled'),
        ('long_only', 'unscaled', 'metrics_lo_unscaled'),
        ('long_only', 'scaled', 'metrics_lo_scaled'),
    ]:
        summary_rows.append(_row(fname, portfolio, scaling, factor_results[fname][mkey]))

for portfolio, scaling, m in [
    ('long_short', 'unscaled', fm_ls_unscaled_m),
    ('long_short', 'scaled', fm_ls_scaled_m),
    ('long_only', 'unscaled', fm_lo_unscaled_m),
    ('long_only', 'scaled', fm_lo_scaled_m),
]:
    summary_rows.append(_row('fm_regression', portfolio, scaling, m))

summary_table = pd.DataFrame(summary_rows)
print('Fama-French Benchmark, EM Universe, Unscaled and vol-targeted')
print(summary_table.to_string(index = False))
print(f'\nFM rank correlation, {fm_rc:.4f}')

# save the consolidated summary for downstream comparison
summary_table.to_csv(results_dir / 'fama_french_summary.csv', index = False)
print('summary saved, fama_french_summary.csv')


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
factor_order = ['value', 'momentum', 'profitability', 'investment', 'size']
# figure 1, volatility targeted cumulative wealth
fig, axes = plt.subplots(1, 2, figsize = (12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + fm_ls_scaled), label = 'FM Long Short')
ax.plot(np.cumprod(1 + fm_lo_scaled), label = 'FM Long Only')
ax.plot(np.cumprod(1 + market_scaled), label = 'Market', linestyle = '--')
ax.set_xlabel('Months from Start of Sample')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Fama and MacBeth Portfolios, Volatility Targeted')
ax.legend(frameon = False)

ax = axes[1]
for factor in factor_order:
    if factor in factor_results:
        ax.plot(
            np.cumprod(1 + factor_results[factor]['returns_lo_scaled']),
            label = factor.title(),
        )
ax.set_xlabel('Months from Start of Sample')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Single Factor Long Only, Volatility Targeted')
ax.legend(frameon = False)

fig.tight_layout()
fig.savefig(results_dir / 'ff_cumulative_scaled.pdf')
fig.savefig(results_dir / 'ff_cumulative_scaled.png')
plt.show()
plt.close(fig)


# figure 2, unscaled cumulative wealth

fig, axes = plt.subplots(1, 2, figsize = (12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + fm_ls_rets), label = 'FM Long Short')
ax.plot(np.cumprod(1 + fm_lo_rets), label = 'FM Long Only')
ax.plot(np.cumprod(1 + np.asarray(market_rets)), label = 'Market', linestyle = '--')
ax.set_xlabel('Months from Start of Sample')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Fama and MacBeth Portfolios, Unscaled')
ax.legend(frameon = False)

ax = axes[1]
for factor in factor_order:
    if factor in factor_results:
        ax.plot(
            np.cumprod(1 + factor_results[factor]['returns_lo_unscaled']),
            label = factor.title(),
        )
ax.set_xlabel('Months')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Single Factor Long Only, Unscaled')
ax.legend(frameon = False)

fig.tight_layout()
fig.savefig(results_dir / 'ff_cumulative_unscaled.pdf')
fig.savefig(results_dir / 'ff_cumulative_unscaled.png')
plt.show()
plt.close(fig)


# figure 3, volatility targeted against unscaled on the same axes.
# scaled and unscaled of the same strategy share a colour so that the pair
# is visually identifiable. the scaled line is solid and the unscaled is dashed.

fig, axes = plt.subplots(1, 2, figsize = (12, 4))

ax = axes[0]
ax.plot(np.cumprod(1 + fm_ls_scaled), label = 'FM Long Short, Scaled', color = 'C0')
ax.plot(np.cumprod(1 + fm_ls_rets), label = 'FM Long Short, Unscaled', color = 'C0', linestyle = '--')
ax.plot(np.cumprod(1 + fm_lo_scaled), label = 'FM Long Only, Scaled', color = 'C1')
ax.plot(np.cumprod(1 + fm_lo_rets), label = 'FM Long Only, Unscaled', color = 'C1', linestyle = '--')
ax.plot(np.cumprod(1 + market_scaled), label = 'Market, Scaled', color = 'C2')
ax.plot(np.cumprod(1 + np.asarray(market_rets)), label = 'Market, Unscaled', color = 'C2', linestyle = '--')
ax.set_xlabel('Months from Start of Sample')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Fama and MacBeth Portfolios, Scaled and Unscaled')
ax.legend(frameon = False, fontsize = 8, loc = 'upper left')

ax = axes[1]
for k, factor in enumerate(factor_order):
    if factor in factor_results:
        col = f'C{k}'
        ax.plot(np.cumprod(1 + factor_results[factor]['returns_lo_scaled']), label = factor.title(), color = col)
        ax.plot(np.cumprod(1 + factor_results[factor]['returns_lo_unscaled']), color = col, linestyle = '--')
ax.set_xlabel('Month')
ax.set_ylabel('Cumulative Wealth')
ax.set_title('Single Factor Long Only, Solid Scaled, Dashed Unscaled')
ax.legend(frameon = False, fontsize = 9, loc = 'upper left')

fig.tight_layout()
fig.savefig(results_dir / 'ff_cumulative_combined.pdf')
fig.savefig(results_dir / 'ff_cumulative_combined.png')
plt.show()
plt.close(fig)
