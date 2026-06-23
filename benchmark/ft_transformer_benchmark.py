"""
Feature Tokeniser Transformer benchmark on the Emerging Markets universe.
"""

import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save
from scipy.stats import spearmanr
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 12


# configuration

train_path = Path('data/processed/train.parquet')
val_path = Path('data/processed/val.parquet')
test_path = Path('data/processed/test.parquet')

results_dir = Path('results/benchmark/ft_transformer_benchmark')
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


# portfolio simulation

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
                'eom': eom, 'id': m['ids'][k],
                'prediction': float(pred[k]), 'realised_return': float(m['r'][k]),
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
        result = spearmanr(pred[valid], m['r'][valid])
        c = float(result[0])  # type: ignore
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def run_quintile_simulation(predictor, month_dates, all_months):
    """
    Quintile simulation producing both the long short and the long only portfolios.
    The long short portfolio holds the top quintile equal weighted long and the
    bottom quintile equal weighted short. The long only portfolio holds the top
    quintile equal weighted with no short leg.
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


# data loading

def load_data():
    """
    Load the three pre processed splits, identify the feature columns from the
    train schema, build the per month rank normalised cross sections, and pool
    the train set into a single tensor.
    Returns a dictionary containing the pooled training tensors, the per month
    cross section dictionary, the feature column list, the date sequences, and
    the boundary dates of the splits.
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
    test_dates = [d for d in sorted_dates if d > val_end]

    x_train = np.vstack([all_months[d]['x'] for d in train_dates])
    y_train = np.concatenate([all_months[d]['r'] for d in train_dates]).astype(np.float32)
    print(f'x_train shape, {x_train.shape}')

    return {
        'x_train': x_train, 'y_train': y_train,
        'all_months': all_months, 'feature_cols': feature_cols,
        'train_dates': train_dates, 'val_dates': val_dates,
        'test_dates': test_dates, 'train_end': train_end,
        'val_end': val_end, 'n_feat': n_feat,
        'n_train_rows': len(train_df), 'n_val_rows': len(val_df),
        'n_test_rows': len(test_df),
    }


def save_training_data(data):
    """
    Save the rank normalised training observations as a csv file with one row
    per firm month and columns for the period end date, the firm identifier,
    the realised excess return, and the full feature set. The split metadata,
    namely the feature column list, the date sequences, and the boundary dates,
    is written separately as a json file.
    """
    feature_cols = data['feature_cols']
    train_dates = data['train_dates']
    val_dates = data['val_dates']
    test_dates = data['test_dates']
    all_months = data['all_months']

    rows = []
    for eom in train_dates:
        m = all_months[eom]
        for k in range(len(m['ids'])):
            row = {
                'eom': eom,
                'id': m['ids'][k],
                'realised_return': float(m['r'][k]),
            }
            for j, col in enumerate(feature_cols):
                row[col] = float(m['x'][k, j])
            rows.append(row)
    train_data_df = pd.DataFrame(rows)
    train_data_df.to_csv(results_dir / 'ft_training_data.csv', index = False)
    print(f'training data saved, ft_training_data.csv, {len(train_data_df):,} rows')

    with open(results_dir / 'ft_training_metadata.json', 'w') as fh:
        json.dump({
            'feature_cols': feature_cols, 'n_features': len(feature_cols),
            'train_dates': [str(d) for d in train_dates], 'val_dates': [str(d) for d in val_dates],
            'test_dates': [str(d) for d in test_dates], 'train_end': str(data['train_end']),
            'val_end': str(data['val_end']), 'n_train_obs': int(data['x_train'].shape[0]),
        }, fh, indent = 2)
    print('training metadata saved, ft_training_metadata.json')


def save_test_inputs(data):
    """
    Save the rank normalised inputs for every firm month in the test set as a
    csv file. The columns mirror those of the training data file and the row
    ordering follows the sorted period end dates.
    """
    feature_cols = data['feature_cols']
    test_dates = data['test_dates']
    all_months = data['all_months']

    rows = []
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
            rows.append(row)
    test_inputs_df = pd.DataFrame(rows)
    test_inputs_df.to_csv(results_dir / 'ft_test_inputs.csv', index = False)
    print(f'test inputs saved, ft_test_inputs.csv, {len(test_inputs_df):,} rows')

    with open(results_dir / 'ft_feature_cols.json', 'w') as fh:
        json.dump({'feature_cols': feature_cols, 'n_features': len(feature_cols)}, fh, indent = 2)
    with open(results_dir / 'ft_splits.json', 'w') as fh:
        json.dump({
            'train_start': str(data['train_dates'][0].date()),
            'train_end': str(data['train_dates'][-1].date()),
            'n_train_months': len(data['train_dates']),
            'val_start': str(data['val_dates'][0].date()),
            'val_end': str(data['val_dates'][-1].date()),
            'n_val_months': len(data['val_dates']),
            'test_start': str(data['test_dates'][0].date()),
            'test_end': str(data['test_dates'][-1].date()),
            'n_test_months': len(data['test_dates']),
        }, fh, indent = 2)


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
        'train_losses': train_losses,
        'val_rank_corr': val_rank_corr,
        'best_epoch': best_epoch,
        'best_val_rc': float(best_rc),
        'n_epochs_run': len(train_losses),
        'd_model': d_model,
        'd_ff': d_ff,
    }


# plotting helpers

def configure_plot_style():
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_cumulative_wealth(ls_returns_raw, ls_returns_scaled, lo_returns_raw, lo_returns_scaled):
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].plot(np.cumprod(1.0 + ls_returns_scaled), label = 'long short vol', color = '#1F4E79')
    axes[0].plot(np.cumprod(1.0 + ls_returns_raw), label = 'long short raw', color = '#1F4E79', linestyle = '--', alpha = 0.5)
    axes[0].set_xlabel('Months from Start of Test Window')
    axes[0].set_ylabel('Cumulative Wealth')
    axes[0].legend(frameon = False, loc = 'upper left')

    axes[1].plot(np.cumprod(1.0 + lo_returns_scaled), label = 'long only vol', color = '#8B2D2D')
    axes[1].plot(np.cumprod(1.0 + lo_returns_raw), label = 'long only raw', color = '#8B2D2D', linestyle = '--', alpha = 0.5)
    axes[1].set_xlabel('Months from Start of Test Window')
    axes[1].set_ylabel('Cumulative Wealth')
    axes[1].legend(frameon = False, loc = 'upper left')

    fig.tight_layout()
    fig.savefig(results_dir / 'ft_cumulative_wealth.pdf')
    fig.savefig(results_dir / 'ft_cumulative_wealth.png')
    plt.close(fig)
    print('cumulative wealth plot saved, ft_cumulative_wealth.pdf and ft_cumulative_wealth.png')


def plot_training_diagnostics(train_losses, val_rank_corr, best_epoch):
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))

    axes[0].plot(train_losses, color = '#1F4E79')
    axes[0].axvline(best_epoch, color = '#8B2D2D', linestyle = '--', alpha = 0.7, label = f'best epoch, {best_epoch}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training MSE')
    axes[0].legend(frameon = False, loc = 'upper right')

    axes[1].plot(val_rank_corr, color = '#1E5F38')
    axes[1].axvline(best_epoch, color = '#8B2D2D', linestyle = '--', alpha = 0.7, label = f'best epoch, {best_epoch}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Rank Correlation')
    axes[1].legend(frameon = False, loc = 'lower right')

    fig.tight_layout()
    fig.savefig(results_dir / 'ft_training_diagnostics.pdf')
    fig.savefig(results_dir / 'ft_training_diagnostics.png')
    plt.close(fig)
    print('training diagnostics plot saved, ft_training_diagnostics.pdf and ft_training_diagnostics.png')


# main routine

def main():
    print(f'torch, {torch.__version__}')
    print(f'cuda available, {cuda_available}')
    print(f'device, {device}')
    if cuda_available:
        print(f'device name, {cuda_device_name}')
        print('mixed precision, enabled')
    print(f'run timestamp utc, {run_timestamp}')
    print(f'train_path, {train_path}')
    print(f'val_path, {val_path}')
    print(f'test_path, {test_path}')
    print(f'results_dir, {results_dir}')

    # load data and save the training data artefacts
    data = load_data()
    save_training_data(data)
    save_test_inputs(data)

    x_train = data['x_train']
    y_train = data['y_train']
    all_months = data['all_months']
    feature_cols = data['feature_cols']
    train_dates = data['train_dates']
    val_dates = data['val_dates']
    test_dates = data['test_dates']
    n_feat = data['n_feat']

    # load the best hyperparameters produced by ft_hpo.py
    best_params_path = results_dir / 'ft_best_params.json'
    if not best_params_path.exists():
        raise FileNotFoundError(
            f'Best parameters file not found at {best_params_path}. '
            f'Run ft_hpo.py first to perform the hyperparameter search.'
        )
    with open(best_params_path) as fh:
        best_data = json.load(fh)
    ft_best = best_data['best_params']
    print(f'best params loaded, {ft_best}')

    # final training with the best hyperparameters
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
    print(f'final model trained in {ft_train_time:.1f} s')
    print(f'parameter count, {n_params:,}')

    # model weights saved as safetensors
    safetensors_save(ft_model.state_dict(), str(results_dir / 'ft_weights.safetensors'))
    print('weights saved, ft_weights.safetensors')

    # training log written as json
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
    print(f'rank corr val, {ft_rc_val:.4f}')
    print(f'rank corr test, {ft_rc_test:.4f}')

    # validation and test predictions saved as csv
    predict_test(ft_predictor, val_dates, all_months).to_csv(results_dir / 'ft_val_predictions.csv', index = False)
    predict_test(ft_predictor, test_dates, all_months).to_csv(results_dir / 'ft_test_predictions.csv', index = False)
    print('val and test predictions saved')

    # test set portfolio simulation
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

    ls['returns_df'].to_csv(results_dir / 'ft_returns_long_short.csv', index = False)
    lo['returns_df'].to_csv(results_dir / 'ft_returns_long_only.csv', index = False)
    ls['holdings_df'].to_csv(results_dir / 'ft_holdings_long_short.csv', index = False)
    lo['holdings_df'].to_csv(results_dir / 'ft_holdings_long_only.csv', index = False)

    mls = ft_metrics['long_short_scaled']
    mlo = ft_metrics['long_only_scaled']
    print(f'long short vol, sharpe = {mls["sharpe"]:.4f}, ann_ret = {mls["ann_ret"] * 100:.2f}%, ann_vol = {mls["ann_vol"] * 100:.2f}%')
    print(f'long only vol, sharpe = {mlo["sharpe"]:.4f}, ann_ret = {mlo["ann_ret"] * 100:.2f}%, ann_vol = {mlo["ann_vol"] * 100:.2f}%')

    # consolidated summary written as json
    summary = {
        'run_timestamp_utc': run_timestamp,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'architecture': {
            'reference': 'Gorishniy et al. (2021)',
            'n_heads': ft_best['n_heads'],
            'd_model': ft_log['d_model'],
            'd_ff': ft_log['d_ff'],
            'n_layers': ft_best['n_layers'],
            'dropout': ft_best['dropout'],
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
            'torch_seed': torch_seed,
        },
        'ft_transformer': {
            'best_params': ft_best,
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
            'portfolio': label, 'sharpe': round(m['sharpe'], 4),
            'ann_ret': round(m['ann_ret'] * 100, 2), 'ann_vol': round(m['ann_vol'] * 100, 2),
            'max_dd': round(m['max_dd'] * 100, 2), 'n_obs': m['n_obs'],
        })
    summary_table = pd.DataFrame(rows)
    print('FT Transformer Benchmark, EM Universe, test set, vol targeted')
    print(summary_table.to_string(index = False))

    # diagnostic plots
    plot_cumulative_wealth(ls['returns'], ls_scaled, lo['returns'], lo_scaled)
    plot_training_diagnostics(ft_log['train_losses'], ft_log['val_rank_corr'], ft_log['best_epoch'])



if __name__ == '__main__':
    main()
