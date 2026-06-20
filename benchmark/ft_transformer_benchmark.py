"""
Feature Tokeniser Transformer benchmark on the Emerging Markets universe.
"""

import gc
import json
import math
import time
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import optuna
from safetensors.torch import save_file as safetensors_save
from scipy.stats import spearmanr
import matplotlib
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 12


# configuration

train_path = Path('data/processed/train.parquet')
val_path = Path('data/processed/val.parquet')
test_path = Path('data/processed/test.parquet')

results_dir = Path('results/benchmark/ft_transformer_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

ret_col = 'ret_exc_lead1m'
rebalance_freq = 3
tc_bps = 25
min_stocks = 30
ret_clip_low = -1.0
ret_clip_high = 1.0

target_vol = 0.10
vol_lookback = 6
max_leverage_ls = 3.0
max_leverage_lo = 3.0

n_epochs_train = 60
patience = 8
grad_clip_norm = 1.0

n_trials = 30
optuna_seed = 24
torch_seed = 48

run_timestamp = datetime.utcnow().isoformat(timespec = 'seconds')


# device
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
use_amp = cuda_available


# model classes
class FeatureTokeniser(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, d_model))
        self.biases = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a = math.sqrt(5))
        nn.init.uniform_(self.biases, -1.0, 1.0)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class FTTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout = dropout, batch_first = True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights = False)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.tokeniser = FeatureTokeniser(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            FTTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        b = x.size(0)
        tokens = self.tokeniser(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim = 1)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.final_norm(tokens[:, 0, :])
        return self.head(cls_out).squeeze(-1)


class FTPredictor:
    def __init__(self, model, device, batch_size = 1024):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def predict(self, x):
        self.model.eval()
        preds = []
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().to(self.device)
            for i in range(0, len(x_t), self.batch_size):
                batch = x_t[i:i + self.batch_size]
                if use_amp:
                    with torch.autocast('cuda'):
                        out = self.model(batch)
                else:
                    out = self.model(batch)
                preds.append(out.float().cpu().numpy())
        return np.concatenate(preds, axis = 0)


def portfolio_metrics(rets):
    rets = np.array(rets, dtype = np.float64)
    if len(rets) == 0:
        return {}
    tw = float((1.0 + rets).prod())
    ann_ret = -1.0 if tw <= 0 else float(tw ** (12.0 / len(rets)) - 1.0)
    ann_vol = float(rets.std() * np.sqrt(12.0))
    sharpe = ann_ret / max(ann_vol, 1e-8)
    se = float(np.sqrt((1.0 + 0.5 * sharpe ** 2) / len(rets)))
    cw = np.cumprod(1.0 + rets)
    pk = np.maximum.accumulate(cw)
    max_dd = float(((pk - cw) / pk).max()) if len(cw) > 0 else 0.0
    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'se_sharpe': se,
        'max_dd': max_dd, 'n_obs': len(rets),
    }


def apply_vol_target(monthly_rets, rebalance_indices, target_vol, vol_lookback, max_leverage):
    scaled = np.array(monthly_rets, dtype = np.float64)
    n = len(monthly_rets)
    n_rb = len(rebalance_indices)
    period_rets = []
    for i in range(1, n_rb):
        window = np.array(monthly_rets[rebalance_indices[i - 1]:rebalance_indices[i]])
        period_rets.append(float(np.prod(1.0 + window) - 1.0))
    for i in range(n_rb):
        if i < vol_lookback:
            continue
        trailing = np.array(period_rets[max(0, i - vol_lookback):i])
        if len(trailing) < 2:
            continue
        sigma_ann = float(trailing.std() * np.sqrt(12.0 / rebalance_freq))
        lev = float(np.clip(target_vol / max(sigma_ann, 1e-8), 1.0 / max_leverage, max_leverage))
        next_rb = rebalance_indices[i + 1] if i + 1 < n_rb else n
        scaled[rebalance_indices[i]:next_rb] = np.array(monthly_rets[rebalance_indices[i]:next_rb]) * lev
    return scaled


def predict_test(predictor, month_dates, all_months):
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


def rank_correlation_oos(predictor, month_dates, all_months):
    corrs = []
    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        pred = predictor.predict(m['x'])
        valid = np.isfinite(pred) & np.isfinite(m['r'])
        if valid.sum() < 10:
            continue
        c, _ = spearmanr(pred[valid], m['r'][valid])
        if not np.isnan(c):
            corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else 0.0


def run_quintile_simulation(predictor, month_dates, all_months):
    """
    Quintile simulation producing both long-short and long-only portfolios.
    Long-short, top quintile equal-weighted long, bottom quintile equal-weighted short.
    Long-only, top quintile equal-weighted, no short leg.
    """
    rset = set(month_dates[::rebalance_freq])

    ls_rets, ls_dates, ls_rb_indices = [], [], []
    lo_rets, lo_dates, lo_rb_indices = [], [], []

    li_ids, si_ids = [], []
    prev_li, prev_si = set(), set()

    ls_holdings, lo_holdings = [], []
    rb_counter = -1

    for eom in month_dates:
        if eom not in all_months:
            continue
        m = all_months[eom]
        ids = m['ids']
        r = m['r']
        x = m['x']

        ls_tcv = 0.0
        lo_tcv = 0.0

        if eom in rset:
            ls_rb_indices.append(len(ls_rets))
            lo_rb_indices.append(len(lo_rets))
            rb_counter += 1

            pred = predictor.predict(x)
            valid = np.isfinite(pred)
            if valid.sum() < 10:
                continue

            vi = ids[valid]
            vp = pred[valid]
            nq = max(1, int(len(vi) * 0.20))
            so = np.argsort(vp)
            li_ids = vi[so[::-1][:nq]].tolist()
            si_ids = vi[so[:nq]].tolist()
            li_set = set(li_ids)
            si_set = set(si_ids)

            ls_to = (len(li_set - prev_li) + len(prev_li - li_set) + len(si_set - prev_si) + len(prev_si - si_set)) / max(nq, 1)
            ls_tcv = ls_to * tc_bps / 10000.0

            lo_to = (len(li_set - prev_li) + len(prev_li - li_set)) / max(nq, 1)
            lo_tcv = lo_to * tc_bps / 10000.0

            prev_li = li_set
            prev_si = si_set

            wt_long = 1.0 / max(len(li_ids), 1)
            wt_short = -1.0 / max(len(si_ids), 1)

            for fid in li_ids:
                ls_holdings.append({
                    'rebalance_index': rb_counter,
                    'eom': eom, 'leg': 'long',
                    'id': fid, 'weight': wt_long,
                })
                lo_holdings.append({
                    'rebalance_index': rb_counter,
                    'eom': eom, 'leg': 'long',
                    'id': fid, 'weight': wt_long,
                })
            for fid in si_ids:
                ls_holdings.append({
                    'rebalance_index': rb_counter,
                    'eom': eom, 'leg': 'short',
                    'id': fid, 'weight': wt_short,
                })

        if not li_ids:
            continue

        li_mask = np.isin(ids, li_ids)
        si_mask = np.isin(ids, si_ids)
        lr = r[li_mask]
        sr = r[si_mask]
        lr_mean = float(lr.mean()) if len(lr) > 0 else 0.0
        sr_mean = float(sr.mean()) if len(sr) > 0 else 0.0

        ls_rets.append(lr_mean - sr_mean - ls_tcv)
        ls_dates.append(eom)
        lo_rets.append(lr_mean - lo_tcv)
        lo_dates.append(eom)

    return {
        'long_short': {
            'returns': np.array(ls_rets),
            'rb_indices': ls_rb_indices,
            'holdings_df': pd.DataFrame(ls_holdings),
            'returns_df': pd.DataFrame({'eom': ls_dates, 'return_raw': ls_rets}),
        },
        'long_only': {
            'returns': np.array(lo_rets),
            'rb_indices': lo_rb_indices,
            'holdings_df': pd.DataFrame(lo_holdings),
            'returns_df': pd.DataFrame({'eom': lo_dates, 'return_raw': lo_rets}),
        },
    }


def train_ft_transformer(params, x_train_pool, y_train_pool, val_dates_local, all_months, n_epochs, patience, device, seed, n_features):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    d_model = params['n_heads'] * params['d_model_per_head']
    d_ff = d_model * params['d_ff_ratio']

    model = FTTransformer(
        n_features = n_features,
        d_model = d_model,
        n_heads = params['n_heads'],
        n_layers = params['n_layers'],
        d_ff = d_ff, dropout = params['dropout'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = params['learning_rate'],
        weight_decay = params['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 3)
    scaler = torch.GradScaler('cuda', enabled = use_amp)
    criterion = nn.MSELoss()

    x_train_t = torch.from_numpy(x_train_pool).float()
    y_train_t = torch.from_numpy(y_train_pool).float()
    if cuda_available:
        x_train_t = x_train_t.pin_memory()
        y_train_t = y_train_t.pin_memory()
    n_train = len(x_train_t)
    batch_size = params['batch_size']

    predictor = FTPredictor(model, device, batch_size = 1024)

    best_rc = -np.inf
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    train_losses = []
    val_rank_corr = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = x_train_t[idx].to(device, non_blocking = True)
            y_batch = y_train_t[idx].to(device, non_blocking = True)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast('cuda'):
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip_norm)
                optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_rc = rank_correlation_oos(predictor, val_dates_local, all_months)
        scheduler.step(val_rc)
        train_losses.append(avg_loss)
        val_rank_corr.append(val_rc)

        if val_rc > best_rc:
            best_rc = val_rc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, {
        'train_losses': train_losses,
        'val_rank_corr': val_rank_corr,
        'best_epoch': best_epoch,
        'best_val_rc': float(best_rc),
        'n_epochs_run': len(train_losses),
        'd_model': d_model,
        'd_ff': d_ff,
    }



def main():
    print(f'device, {device}')
    if cuda_available:
        print(f'device name, {cuda_device_name}')
        print(f'mixed precision, enabled')
    print(f'run timestamp utc, {run_timestamp}')
    print(f'train_path, {train_path}')
    print(f'val_path, {val_path}')
    print(f'test_path, {test_path}')
    print(f'results_dir, {results_dir}')

    # feature selection from train schema
    train_schema = pq.read_schema(train_path)
    non_feature = {
        'id', 'gvkey', 'isin', 'cusip', 'permno', 'permco',
        'eom', 'excntry', 'sic', 'naics', 'source_crsp', ret_col,
    }
    feature_cols = [
        c for c in train_schema.names
        if c not in non_feature
        and pa.types.is_floating(train_schema.field(c).type)
        and '_lag' not in c
    ]
    print(f'feature columns selected, {len(feature_cols)}')

    # load the pre-processed splits
    needed = list(dict.fromkeys(
        [c for c in ['id', 'eom', 'excntry', ret_col] + feature_cols if c in train_schema.names]
    ))
    train_df = pd.read_parquet(train_path, columns = needed)
    val_df = pd.read_parquet(val_path, columns = needed)
    test_df = pd.read_parquet(test_path, columns = needed)

    for d in (train_df, val_df, test_df):
        d['eom'] = pd.to_datetime(d['eom'])

    train_end = train_df['eom'].max()
    val_end = val_df['eom'].max()

    df = pd.concat([train_df, val_df, test_df], axis = 0, ignore_index = True)
    for col in feature_cols:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    df[ret_col] = df[ret_col].clip(lower = ret_clip_low, upper = ret_clip_high)

    print(f'train rows, {len(train_df):,}')
    print(f'val rows, {len(val_df):,}')
    print(f'test rows, {len(test_df):,}')
    print(f'combined rows, {len(df):,}')

    # build per-month cross-sections with rank normalisation
    sorted_eoms = sorted(df['eom'].unique())
    all_months = {}
    n_feat = len(feature_cols)

    for eom in sorted_eoms:
        month = df[df['eom'] == eom].copy()
        month = month[month[ret_col].notna()]
        if len(month) < min_stocks:
            continue
        ids = month['id'].values
        r = month[ret_col].values.astype(np.float64)
        x = np.zeros((len(month), n_feat), dtype = np.float32)
        for j, col in enumerate(feature_cols):
            if col not in month.columns:
                continue
            vals = month[col].values.astype(np.float64)
            valid = np.isfinite(vals)
            if valid.sum() > 1:
                x[valid, j] = (pd.Series(vals[valid]).rank(pct = True).values - 0.5).astype(np.float32)
        all_months[eom] = {'ids': ids, 'r': r, 'x': x}

    sorted_dates = sorted(all_months.keys())
    print(f'processed months, {len(sorted_dates)}')

    # date based split
    train_dates = [d for d in sorted_dates if d <= train_end]
    val_dates = [d for d in sorted_dates if train_end < d <= val_end]
    test_dates = [d for d in sorted_dates if d > val_end]

    x_train = np.vstack([all_months[d]['x'] for d in train_dates])
    y_train = np.concatenate([all_months[d]['r'] for d in train_dates]).astype(np.float32)
    print(f'x_train shape, {x_train.shape}')

    # save training data to disk for reproducibility
    np.savez_compressed(
        results_dir / 'ft_training_data.npz',
        x_train = x_train,
        y_train = y_train,
        feature_cols = np.array(feature_cols),
        train_dates = np.array([str(d) for d in train_dates]),
        val_dates = np.array([str(d) for d in val_dates]),
        test_dates = np.array([str(d) for d in test_dates]),
        train_end = str(train_end),
        val_end = str(val_end),
    )
    print(f'training data saved, ft_training_data.npz')

    # save the per month rank-normalised inputs for the test set as parquet
    test_inputs_rows = []
    for eom in test_dates:
        m = all_months[eom]
        for k in range(len(m['ids'])):
            row = {
                'eom': eom,
                'id': m['ids'][k],
                'realised_return': float(m['r'][k]),
            }
            for j, col in enumerate(feature_cols):
                row[col] = float(m['x'][k, j])
            test_inputs_rows.append(row)
    test_inputs_df = pd.DataFrame(test_inputs_rows)
    test_inputs_df.to_parquet(results_dir / 'ft_test_inputs.parquet', index = False)
    print(f'test inputs saved, ft_test_inputs.parquet, {len(test_inputs_df):,} rows')

    # save feature column list and split boundaries as separate json for easy reading
    with open(results_dir / 'ft_feature_cols.json', 'w') as fh:
        json.dump({'feature_cols': feature_cols, 'n_features': len(feature_cols)}, fh, indent = 2)
    with open(results_dir / 'ft_splits.json', 'w') as fh:
        json.dump({
            'train_start': str(train_dates[0].date()),
            'train_end': str(train_dates[-1].date()),
            'n_train_months': len(train_dates),
            'val_start': str(val_dates[0].date()),
            'val_end': str(val_dates[-1].date()),
            'n_val_months': len(val_dates),
            'test_start': str(test_dates[0].date()),
            'test_end': str(test_dates[-1].date()),
            'n_test_months': len(test_dates),
        }, fh, indent = 2)

    # hyperparameter search
    def ft_objective(trial):
        params = {
            'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
            'd_model_per_head': trial.suggest_categorical('d_model_per_head', [8, 16, 32]),
            'd_ff_ratio': trial.suggest_categorical('d_ff_ratio', [2, 4]),
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log = True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log = True),
        }
        model, log = train_ft_transformer(
            params = params,
            x_train_pool = x_train,
            y_train_pool = y_train,
            val_dates_local = val_dates,
            all_months = all_months,
            n_epochs = n_epochs_train,
            patience = patience,
            device = device,
            seed = torch_seed,
            n_features = n_feat,
        )
        predictor = FTPredictor(model, device, batch_size = 1024)
        sim = run_quintile_simulation(predictor, val_dates, all_months)
        ls, lo = sim['long_short'], sim['long_only']
        if len(ls['returns']) == 0:
            return -999.0
        ls_scaled = apply_vol_target(ls['returns'], ls['rb_indices'], target_vol, vol_lookback, max_leverage_ls)
        lo_scaled = apply_vol_target(lo['returns'], lo['rb_indices'], target_vol, vol_lookback, max_leverage_lo)
        ls_sharpe = portfolio_metrics(ls_scaled).get('sharpe', -999.0)
        lo_sharpe = portfolio_metrics(lo_scaled).get('sharpe', -999.0)

        trial.set_user_attr('best_epoch', log['best_epoch'])
        trial.set_user_attr('n_epochs_run', log['n_epochs_run'])
        trial.set_user_attr('best_val_rc', log['best_val_rc'])
        trial.set_user_attr('val_sharpe_long_only', float(lo_sharpe))
        trial.set_user_attr('d_model', log['d_model'])
        trial.set_user_attr('d_ff', log['d_ff'])
        trial.set_user_attr('param_count', sum(p.numel() for p in model.parameters()))

        del model, predictor
        gc.collect()
        if cuda_available:
            torch.cuda.empty_cache()
        return ls_sharpe

    ft_study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
        study_name = f'ft_{run_timestamp}',
    )
    t0 = time.time()
    ft_study.optimize(ft_objective, n_trials = n_trials, show_progress_bar = True)
    ft_hpo_time = time.time() - t0
    ft_best = ft_study.best_params
    print(f'FT best val ls sharpe, {ft_study.best_value:.4f}')
    print(f'FT best params, {ft_best}')
    print(f'FT hpo time, {ft_hpo_time:.1f} s')

    # save Optuna study and trial history
    ft_trials_df = ft_study.trials_dataframe()
    ft_trials_df.to_csv(results_dir / 'ft_optuna_trials.csv', index = False)
    with open(results_dir / 'ft_optuna_study.pkl', 'wb') as fh:
        pickle.dump(ft_study, fh)
    with open(results_dir / 'ft_best_params.json', 'w') as fh:
        json.dump({
            'best_params': ft_best,
            'best_val_long_short_sharpe': float(ft_study.best_value),
            'best_trial_number': int(ft_study.best_trial.number),
            'best_trial_user_attrs': dict(ft_study.best_trial.user_attrs),
        }, fh, indent = 2, default = float)
    print(f'optuna study saved, {len(ft_trials_df)} trials')

    # final training with best params
    t0 = time.time()
    ft_model, ft_log = train_ft_transformer(
        params = ft_best,
        x_train_pool = x_train,
        y_train_pool = y_train,
        val_dates_local = val_dates,
        all_months = all_months,
        n_epochs = n_epochs_train,
        patience = patience,
        device = device,
        seed = torch_seed,
        n_features = n_feat,
    )
    ft_train_time = time.time() - t0
    ft_predictor = FTPredictor(ft_model, device, batch_size = 1024)
    n_params = sum(p.numel() for p in ft_model.parameters())
    print(f'FT final model trained in {ft_train_time:.1f} s')
    print(f'parameter count, {n_params:,}')

    # save model weights as safetensors
    safetensors_save(ft_model.state_dict(), str(results_dir / 'ft_weights.safetensors'))
    print(f'weights saved, ft_weights.safetensors')

    # save training log
    ft_train_log = {
        'train_losses': ft_log['train_losses'],
        'val_rank_corr': ft_log['val_rank_corr'],
        'best_epoch': ft_log['best_epoch'],
        'best_val_rc': ft_log['best_val_rc'],
        'n_epochs_run': ft_log['n_epochs_run'],
        'd_model': ft_log['d_model'],
        'd_ff': ft_log['d_ff'],
        'training_time_seconds': float(ft_train_time),
        'parameter_count': int(n_params),
    }
    with open(results_dir / 'ft_train_log.json', 'w') as fh:
        json.dump(ft_train_log, fh, indent = 2, default = float)

    # validation diagnostics
    ft_rc_val = rank_correlation_oos(ft_predictor, val_dates, all_months)
    ft_rc_test = rank_correlation_oos(ft_predictor, test_dates, all_months)
    print(f'FT rank corr val, {ft_rc_val:.4f}')
    print(f'FT rank corr test, {ft_rc_test:.4f}')

    # validation predictions saved
    val_predictions = predict_test(ft_predictor, val_dates, all_months)
    val_predictions.to_parquet(results_dir / 'ft_val_predictions.parquet', index = False)

    # test simulation
    sim = run_quintile_simulation(ft_predictor, test_dates, all_months)
    ls, lo = sim['long_short'], sim['long_only']

    ls_scaled = apply_vol_target(ls['returns'], ls['rb_indices'], target_vol, vol_lookback, max_leverage_ls)
    lo_scaled = apply_vol_target(lo['returns'], lo['rb_indices'], target_vol, vol_lookback, max_leverage_lo)
    ls['returns_df']['return_scaled'] = ls_scaled
    lo['returns_df']['return_scaled'] = lo_scaled

    ft_metrics = {
        'long_short_raw': portfolio_metrics(ls['returns']),
        'long_short_scaled': portfolio_metrics(ls_scaled),
        'long_only_raw': portfolio_metrics(lo['returns']),
        'long_only_scaled': portfolio_metrics(lo_scaled),
    }

    # save returns, holdings, and predictions for the test set
    ls['returns_df'].to_parquet(results_dir / 'ft_returns_long_short.parquet', index = False)
    lo['returns_df'].to_parquet(results_dir / 'ft_returns_long_only.parquet', index = False)
    ls['holdings_df'].to_parquet(results_dir / 'ft_holdings_long_short.parquet', index = False)
    lo['holdings_df'].to_parquet(results_dir / 'ft_holdings_long_only.parquet', index = False)
    predict_test(ft_predictor, test_dates, all_months).to_parquet(results_dir / 'ft_test_predictions.parquet', index = False)

    mls = ft_metrics['long_short_scaled']
    mlo = ft_metrics['long_only_scaled']
    print(f'FT long-short vol, sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
    print(f'FT long-only vol, sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')

    # consolidated summary
    summary = {
        'run_timestamp_utc': run_timestamp,
        'universe': 'EM',
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'architecture': {
            'name': 'Feature Tokeniser Transformer',
            'reference': 'Gorishniy et al. (2021)',
            'n_heads': ft_best['n_heads'],
            'd_model': ft_log['d_model'],
            'd_ff': ft_log['d_ff'],
            'n_layers': ft_best['n_layers'],
            'dropout': ft_best['dropout'],
            'activation': 'GELU',
            'normalisation': 'Pre-LN',
            'pooling': 'CLS token',
            'task': 'regression (ret_exc_lead1m)',
            'loss': 'MSE',
            'imputation': 'median (zero in rank-normalised space)',
            'parameter_count': int(n_params),
        },
        'split': {
            'train': {
                'start': str(train_dates[0].date()),
                'end': str(train_dates[-1].date()),
                'n_months': len(train_dates),
                'n_obs': int(x_train.shape[0]),
            },
            'val': {
                'start': str(val_dates[0].date()),
                'end': str(val_dates[-1].date()),
                'n_months': len(val_dates),
            },
            'test': {
                'start': str(test_dates[0].date()),
                'end': str(test_dates[-1].date()),
                'n_months': len(test_dates),
            },
        },
        'config': {
            'rebalance_freq': rebalance_freq,
            'tc_bps': tc_bps,
            'min_stocks': min_stocks,
            'ret_clip': [ret_clip_low, ret_clip_high],
            'target_vol': target_vol,
            'vol_lookback': vol_lookback,
            'max_leverage_ls': max_leverage_ls,
            'max_leverage_lo': max_leverage_lo,
            'n_epochs_train': n_epochs_train,
            'patience': patience,
            'grad_clip_norm': grad_clip_norm,
            'optuna_seed': optuna_seed,
            'torch_seed': torch_seed,
            'n_trials': n_trials,
        },
        'ft_transformer': {
            'best_params': ft_best,
            'best_val_long_short_sharpe': float(ft_study.best_value),
            'best_trial_number': int(ft_study.best_trial.number),
            'n_trials_completed': sum(1 for t in ft_study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(ft_hpo_time),
            'final_training_time_seconds': float(ft_train_time),
            'best_epoch': ft_log['best_epoch'],
            'n_epochs_run': ft_log['n_epochs_run'],
            'rc_val': float(ft_rc_val),
            'rc_test': float(ft_rc_test),
            'portfolio_metrics': ft_metrics,
        },
    }
    with open(results_dir / 'ft_summary.json', 'w') as fh:
        json.dump(summary, fh, indent = 2, default = float)
    print('summary saved')

    # results table
    rows = []
    for label, key in [('long_short', 'long_short_scaled'), ('long_only', 'long_only_scaled')]:
        m = ft_metrics[key]
        rows.append({
            'portfolio': label,
            'sharpe': round(m['sharpe'], 4),
            'ann_ret': round(m['ann_ret'] * 100, 2),
            'ann_vol': round(m['ann_vol'] * 100, 2),
            'max_dd': round(m['max_dd'] * 100, 2),
            'n_obs': m['n_obs'],
        })
    summary_table = pd.DataFrame(rows)
    print('\nFT-Transformer Benchmark, EM Universe, test set, vol-targeted')
    print(summary_table.to_string(index = False))

    # image 1, cumulative wealth for long-short and long-only portfolios.
    # image 2, training diagnostics, namely loss and validation rank correlation.
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # image 1, cumulative wealth
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].plot(np.cumprod(1.0 + ls_scaled), label = 'long-short vol', color = '#1F4E79')
    axes[0].plot(np.cumprod(1.0 + ls['returns']), label = 'long-short raw', color = '#1F4E79', linestyle = '--', alpha = 0.5)
    axes[0].set_xlabel('Months from Start of Test Window')
    axes[0].set_ylabel('Cumulative Wealth')
    axes[0].legend(frameon = False, loc = 'upper left')

    axes[1].plot(np.cumprod(1.0 + lo_scaled), label = 'long-only vol', color = '#8B2D2D')
    axes[1].plot(np.cumprod(1.0 + lo['returns']), label = 'long-only raw', color = '#8B2D2D', linestyle = '--', alpha = 0.5)
    axes[1].set_xlabel('Months from Start of Test Window')
    axes[1].set_ylabel('Cumulative Wealth')
    axes[1].legend(frameon = False, loc = 'upper left')

    fig.tight_layout()
    fig.savefig(results_dir / 'ft_cumulative_wealth.pdf')
    fig.savefig(results_dir / 'ft_cumulative_wealth.png')
    plt.close(fig)
    print('cumulative wealth plot saved, ft_cumulative_wealth.pdf and .png')

    # image 2, training diagnostics
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].plot(ft_log['train_losses'], color = '#1F4E79')
    axes[0].axvline(ft_log['best_epoch'], color = '#8B2D2D', linestyle = '--', alpha = 0.7, label = f'best epoch, {ft_log["best_epoch"]}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training MSE')
    axes[0].legend(frameon = False, loc = 'upper right')

    axes[1].plot(ft_log['val_rank_corr'], color = '#1E5F38')
    axes[1].axvline(ft_log['best_epoch'], color = '#8B2D2D', linestyle = '--', alpha = 0.7, label = f'best epoch, {ft_log["best_epoch"]}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Rank Correlation')
    axes[1].legend(frameon = False, loc = 'lower right')

    fig.tight_layout()
    fig.savefig(results_dir / 'ft_training_diagnostics.pdf')
    fig.savefig(results_dir / 'ft_training_diagnostics.png')
    plt.close(fig)



if __name__ == '__main__':
    main()
