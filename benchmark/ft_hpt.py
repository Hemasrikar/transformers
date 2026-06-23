"""
Hyperparameter search for the Feature Tokeniser Transformer benchmark on the
Emerging Markets universe.

The search procedure is a Tree structured Parzen Estimator over the
architectural and optimisation hyperparameters of the Feature Tokeniser
Transformer. The objective is the validation long short Sharpe ratio at
the best rank correlation epoch, evaluated on the same volatility targeted
quintile simulation that is used for the final evaluation in ft_model.py.
The validation long only Sharpe ratio is recorded as a trial user
attribute.

The selected best hyperparameters are written to ft_best_params.json,
which ft_model.py reads at its main entry point in order to train the
final model.
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
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# configuration

train_path = Path('data/processed/train.parquet')
val_path = Path('data/processed/val.parquet')
test_path = Path('data/processed/test.parquet')

results_dir = Path('../results/benchmark/ft_transformer_benchmark')
results_dir.mkdir(parents = True, exist_ok = True)

ret_col = 'ret_exc_lead1m'
rebalance_freq = 6
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
torch_seed = 24

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


# portfolio simulation helpers

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
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'se_sharpe': se,
        'max_dd': max_dd,
        'n_obs': len(rets),
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
        result = spearmanr(pred[valid], m['r'][valid])
        c = float(result[0])  # type: ignore
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def run_quintile_simulation(predictor, month_dates, all_months):
    """
    Quintile simulation that returns the long short and long only return
    series in a single pass. The long short portfolio holds the top
    quintile equal weighted long and the bottom quintile equal weighted
    short. The long only portfolio holds the top quintile equal weighted
    with no short leg.
    """
    rset = set(month_dates[::rebalance_freq])

    ls_rets, ls_rb_indices = [], []
    lo_rets, lo_rb_indices = [], []

    li_ids, si_ids = [], []
    prev_li, prev_si = set(), set()

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

        if not li_ids:
            continue

        li_mask = np.isin(ids, li_ids)
        si_mask = np.isin(ids, si_ids)
        lr = r[li_mask]
        sr = r[si_mask]
        lr_mean = float(lr.mean()) if len(lr) > 0 else 0.0
        sr_mean = float(sr.mean()) if len(sr) > 0 else 0.0

        ls_rets.append(lr_mean - sr_mean - ls_tcv)
        lo_rets.append(lr_mean - lo_tcv)

    return {
        'long_short': {'returns': np.array(ls_rets), 'rb_indices': ls_rb_indices},
        'long_only': {'returns': np.array(lo_rets), 'rb_indices': lo_rb_indices},
    }


# data loading

def load_data():
    """
    Load the three pre processed splits, identify the feature columns from
    the train schema, build the per month rank normalised cross sections,
    and pool the train set into a single tensor. Returns a dictionary that
    contains the pooled training tensors, the per month cross section
    dictionary, the feature column list, and the validation dates that the
    search objective requires.
    """
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
                ranked = np.asarray(pd.Series(vals[valid]).rank(pct = True).values, dtype = np.float64)
                x[valid, j] = (ranked - 0.5).astype(np.float32)
        all_months[eom] = {'ids': ids, 'r': r, 'x': x}

    sorted_dates = sorted(all_months.keys())
    train_dates = [d for d in sorted_dates if d <= train_end]
    val_dates = [d for d in sorted_dates if train_end < d <= val_end]

    x_train = np.vstack([all_months[d]['x'] for d in train_dates])
    y_train = np.concatenate([all_months[d]['r'] for d in train_dates]).astype(np.float32)
    print(f'x_train shape, {x_train.shape}')

    return {
        'x_train': x_train, 'y_train': y_train,
        'all_months': all_months, 'feature_cols': feature_cols,
        'val_dates': val_dates, 'n_feat': n_feat,
    }


# training function

def train_ft_transformer(params, x_train_pool, y_train_pool, val_dates_local, all_months, n_epochs, patience, device, seed, n_features):
    """
    Train a Feature Tokeniser Transformer with the given hyperparameters and
    early stopping on the validation rank correlation. Returns the trained
    model and a dictionary of training diagnostics.
    """
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
        d_ff = d_ff,
        dropout = params['dropout'],
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
        'train_losses': train_losses, 'val_rank_corr': val_rank_corr,
        'best_epoch': best_epoch, 'best_val_rc': float(best_rc),
        'n_epochs_run': len(train_losses), 'd_model': d_model,
        'd_ff': d_ff,
    }


# main routine

def main():
    print(f'device, {device}')
    if cuda_available:
        print(f'device name, {cuda_device_name}')
        print('mixed precision, enabled')
    print(f'run timestamp utc, {run_timestamp}')
    print(f'results_dir, {results_dir}')
    print(f'n_trials, {n_trials}')

    # load data once and reuse the cross sections across every trial
    data = load_data()
    x_train = data['x_train']
    y_train = data['y_train']
    all_months = data['all_months']
    val_dates = data['val_dates']
    n_feat = data['n_feat']

    # objective function defined as a closure over the loaded data
    def objective(trial):
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

    # create the study and run the search
    study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = optuna_seed),
        study_name = f'ft_em_{run_timestamp}',
    )
    t0 = time.time()
    study.optimize(objective, n_trials = n_trials, show_progress_bar = True)
    hpo_time = time.time() - t0
    best_params = study.best_params

    print(f'best val ls sharpe, {study.best_value:.4f}')
    print(f'best params, {best_params}')
    print(f'hpo time, {hpo_time:.1f} s, {hpo_time / 60:.2f} min')

    # save the trial history as csv and the study object as pickle
    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_dir / 'ft_optuna_trials.csv', index = False)
    with open(results_dir / 'ft_optuna_study.pkl', 'wb') as fh:
        pickle.dump(study, fh)

    # save the best hyperparameters as json
    with open(results_dir / 'ft_best_params.json', 'w') as fh:
        json.dump({
            'best_params': best_params,
            'best_val_long_short_sharpe': float(study.best_value),
            'best_trial_number': int(study.best_trial.number),
            'best_trial_user_attrs': dict(study.best_trial.user_attrs),
            'n_trials_completed': sum(1 for t in study.trials if t.state.name == 'COMPLETE'),
            'hpo_time_seconds': float(hpo_time),
            'optuna_seed': optuna_seed,
            'torch_seed': torch_seed,
            'run_timestamp_utc': run_timestamp,
        }, fh, indent = 2, default = float)
    print('best params saved, ft_best_params.json')
    print('optuna trials saved, ft_optuna_trials.csv')
    print('optuna study saved, ft_optuna_study.pkl')


if __name__ == '__main__':
    main()
