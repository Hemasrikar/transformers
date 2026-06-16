"""
Nonlinear Portfolio Transformer: Encoding Variant Comparison

"""

import gc
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')



# Configuration

country = 'EM'

# Path to raw JKP parquet file. Note the space in the filename.
raw_path = Path(f'data/Global Factor_{country}.parquet')

results_dir = Path('results') / country

# Data processing
coverage_threshold = 0.70
max_miss_frac = 1.0 / 3.0
min_stocks = 30

# Train / val / test split dates (inclusive upper bounds)
train_end = '2014-12-31'
val_end = '2019-12-31'
# test is everything after val_end

# Variants to train. Remove entries to skip specific variants.
variant_list = ['identity', 'linear', 'ple', 'periodic', 'fourier', 'magnitude_dir']

# Architecture
n_blocks = 2
n_heads = 1
d_ff = 256

# Training
n_epochs = 50
lr = 1e-5
weight_decay = 1e-3
grad_clip = 1.0
n_seeds = 3
patience = 10

# Portfolio simulation
rebalance_freq = 6    # months
tc_bps = 25   # one-way transaction cost in basis points

# Columns never treated as characteristics
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




# Data processing

def process_raw_data():

	# Load
	schema = pq.read_schema(raw_path)
	char_candidate = [c for c in schema.names if c not in exclude_cols and c not in load_always]
	needed = [c for c in load_always + char_candidate if c in schema.names]
	df = pd.read_parquet(raw_path, columns = needed)
	df['eom'] = pd.to_datetime(df['eom'])
	print(f'loaded: {df.shape[0]:,} rows, {df["eom"].min().date()} to {df["eom"].max().date()}')

	# Cast to float32
	for col in char_candidate:
		if col in df.columns and df[col].dtype == np.float64:
			df[col] = df[col].astype(np.float32)
	if 'me' in df.columns and df['me'].dtype == np.float64:
		df['me'] = df['me'].astype(np.float32)

	char_candidate = [c for c in char_candidate if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
	print(f'numeric candidates: {len(char_candidate)}')

	# Coverage filter on training period only
	df_tr = df[df['eom'] <= train_end]
	coverage = df_tr[char_candidate].notna().mean()
	char_cols = sorted([c for c in char_candidate if coverage[c] >= coverage_threshold])
	d = len(char_cols)
	print(f'features with >= {coverage_threshold:.0%} coverage: d = {d}')
	del df_tr

	# Keep only needed columns
	id_cols = [c for c in load_always if c in df.columns]
	df = df[id_cols + char_cols]

	# Missing filter
	n_miss = df[char_cols].isna().sum(axis = 1)
	df = df[n_miss <= d * max_miss_frac].reset_index(drop = True)
	print(f'after missing filter: {len(df):,} rows')

	# Firm lookup
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
	id_to_gvkey = dict(zip(firm_lookup['id'], firm_lookup.get('gvkey', firm_lookup['id'])))

	# Per-month rank normalisation to [-0.5, 0.5]
	all_months = {}
	for eom in sorted(df['eom'].unique()):
		month = df[df['eom'] == eom].copy()
		if len(month) < min_stocks:
			continue
		ranks = month[char_cols].rank(pct = True, axis = 0) - 0.5
		month[char_cols] = ranks.fillna(0.0)
		x = month[char_cols].values.astype(np.float32)
		r = month['ret_exc_lead1m'].values.astype(np.float32)
		ids = month['id'].values
		hr = np.isfinite(r)
		if hr.sum() >= 5:
			all_months[eom] = {'x': x, 'r': r, 'has_ret': hr, 'ids': ids}

	del df
	gc.collect()

	# Date-based split
	train_months = {eom: m for eom, m in all_months.items() if eom <= pd.Timestamp(train_end)}
	val_months = {eom: m for eom, m in all_months.items() if pd.Timestamp(train_end) < eom <= pd.Timestamp(val_end)}
	test_months = {eom: m for eom, m in all_months.items() if eom > pd.Timestamp(val_end)}

	print(f'train: {len(train_months)} months, val: {len(val_months)}, test: {len(test_months)}')
	print(f'mean firms per month: {np.mean([m["x"].shape[0] for m in all_months.values()]):.0f}')

	return train_months, val_months, test_months, char_cols, d, id_to_gvkey


def to_gpu(md, device):
	result = {}
	for eom, m in md.items():
		hr = m['has_ret']
		if hr.sum() < 5:
			continue
		result[eom] = {
			'x': torch.tensor(m['x'][hr], dtype = torch.float32, device = device),
			'r': torch.tensor(m['r'][hr], dtype = torch.float32, device = device),
			'ids': m['ids'][hr],
		}
	return result


# Encoding variants

class IdentityEncoder(nn.Module):
	def forward(self, x):
		return x

class LinearEncoder(nn.Module):
	def __init__(self, n):
		super().__init__()
		self.w = nn.Parameter(torch.ones(n))
		self.b = nn.Parameter(torch.zeros(n))
	def forward(self, x):
		return x * self.w + self.b

class PLEEncoder(nn.Module):
	def __init__(self, n, bins = 16):
		super().__init__()
		bd = torch.linspace(-0.5, 0.5, bins + 1)
		self.register_buffer('lo', bd[:-1])
		self.register_buffer('hi', bd[1:])
		self.w = nn.Parameter(torch.zeros(n, bins))
	def forward(self, x):
		a = torch.clamp((x.unsqueeze(-1) - self.lo) / (self.hi - self.lo + 1e-8), 0, 1)
		return x + (a * self.w.unsqueeze(0)).sum(-1)

class PeriodicEncoder(nn.Module):
	def __init__(self, n, nf = 8):
		super().__init__()
		self.om = nn.Parameter(torch.randn(n, nf))
		self.ph = nn.Parameter(torch.randn(n, nf) * 0.1)
		self.c = nn.Parameter(torch.zeros(n, nf))
	def forward(self, x):
		return x + (torch.sin(x.unsqueeze(-1) * self.om.unsqueeze(0) + self.ph.unsqueeze(0)) * self.c.unsqueeze(0)).sum(-1)

class FourierEncoder(nn.Module):
	def __init__(self, n, nf = 8):
		super().__init__()
		self.register_buffer('freq', torch.arange(1, nf + 1, dtype = torch.float32) * torch.pi)
		self.a = nn.Parameter(torch.zeros(n, nf))
		self.b = nn.Parameter(torch.zeros(n, nf))
	def forward(self, x):
		s = x.unsqueeze(-1) * self.freq
		return x + (torch.sin(s) * self.a.unsqueeze(0) + torch.cos(s) * self.b.unsqueeze(0)).sum(-1)

class MagnitudeDirectionEncoder(nn.Module):
	def __init__(self, n):
		super().__init__()
		self.wp = nn.Parameter(torch.ones(n))
		self.wn = nn.Parameter(torch.ones(n))
		self.b = nn.Parameter(torch.zeros(n))
	def forward(self, x):
		return F.relu(x) * self.wp - F.relu(-x) * self.wn + self.b

def build_encoder(v, n):
	enc = {
		'identity': IdentityEncoder, 'linear': LinearEncoder, 'ple': PLEEncoder,
		'periodic': PeriodicEncoder, 'fourier': FourierEncoder, 'magnitude_dir': MagnitudeDirectionEncoder,
	}
	return enc[v]() if v == 'identity' else enc[v](n)


# Architecture

class AttentionHead(nn.Module):
	def __init__(self, n, s):
		super().__init__()
		self.w = nn.Parameter(torch.randn(n, n) * s)
		self.v = nn.Parameter(torch.randn(n, n) * s)
		self.sc = 1.0 / np.sqrt(n)
	def forward(self, y):
		return F.softmax((y @ self.w @ y.t()) * self.sc, dim = -1) @ (y @ self.v)

class TransformerBlock(nn.Module):
	def __init__(self, n, h, ff, s):
		super().__init__()
		self.heads = nn.ModuleList([AttentionHead(n, s) for _ in range(h)])
		self.w1 = nn.Parameter(torch.randn(n, ff) * (1.0 / ff))
		self.b1 = nn.Parameter(torch.zeros(ff))
		self.w2 = nn.Parameter(torch.randn(ff, n) * s)
		self.b2 = nn.Parameter(torch.zeros(n))
	def forward(self, y):
		y = sum(h(y) for h in self.heads) + y
		return F.relu(y @ self.w1 + self.b1) @ self.w2 + self.b2 + y

class PortfolioTransformer(nn.Module):
	def __init__(self, n, nb, nh, ff, enc):
		super().__init__()
		self.enc = enc
		s = 1.0 / n
		self.blocks = nn.ModuleList([TransformerBlock(n, nh, ff, s) for _ in range(nb)])
		self.lam    = nn.Parameter(torch.randn(n) * s)
	def forward(self, x):
		y = self.enc(x)
		for b in self.blocks:
			y = b(y)
		return y @ self.lam
	def msrr_loss(self, x, r):
		return (1.0 - self.forward(x) @ r) ** 2


# Training

@torch.no_grad()
def eval_rank_corr(model, gpu_months):
	model.eval()
	corrs = []
	for m in gpu_months.values():
		w = model(m['x']).cpu().numpy()
		r = m['r'].cpu().numpy()
		if len(w) < 10:
			continue
		c, _ = spearmanr(w, r)
		if not np.isnan(c):
			corrs.append(c)
	model.train()
	return float(np.mean(corrs)) if corrs else 0.0


@torch.no_grad()
def predict_all(model, gpu_months):
	model.eval()
	return {
		eom: {'w': model(m['x']).cpu().numpy(), 'ids': m['ids'], 'r': m['r'].cpu().numpy()}
		for eom, m in gpu_months.items()
	}


def train_one_seed(variant, seed, device):
	torch.manual_seed(seed)
	np.random.seed(seed)
	model = PortfolioTransformer(d, n_blocks, n_heads, d_ff, build_encoder(variant, d).to(device)).to(device)
	opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
	keys = list(train_gpu.keys())
	bv, be, bs, wait = -np.inf, 0, None, 0

	for ep in range(1, n_epochs + 1):
		model.train()
		for idx in np.random.permutation(len(keys)):
			opt.zero_grad()
			loss = model.msrr_loss(train_gpu[keys[idx]]['x'], train_gpu[keys[idx]]['r'])
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
		vc = eval_rank_corr(model, val_gpu)
		if vc > bv:
			bv, be = vc, ep
			bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
			wait = 0
		else:
			wait += 1
		if wait >= patience:
			break

	if bs:
		model.load_state_dict(bs)
		model.to(device)
	return model, be, bv


def train_variant(variant, device):
	print(f'\nVariant: {variant}')
	vdir = results_dir / variant
	vdir.mkdir(parents = True, exist_ok = True)
	t0 = time.time()

	all_sw = []
	for seed in range(n_seeds):
		model, be, bv = train_one_seed(variant, seed, device)
		print(f'  seed {seed}: epoch {be}, val corr {bv:.4f}')
		sys.stdout.flush()
		torch.save(model.state_dict(), vdir / f'{variant}_{country}_seed{seed}.pt')
		all_sw.append(predict_all(model, {**train_gpu, **val_gpu, **test_gpu}))
		del model
		gc.collect()
		torch.cuda.empty_cache()

	averaged = {}
	for eom in all_sw[0]:
		ws, sdf_s, nv = np.zeros(len(all_sw[0][eom]['w'])), 0.0, 0
		for sw in all_sw:
			w = sw[eom]['w'].astype(np.float64)
			a = np.abs(w).sum()
			if a > 1e-10:
				ws += w / a
				sdf_s += float(w @ sw[eom]['r'].astype(np.float64))
				nv += 1
		if nv > 0:
			averaged[eom] = {
				'w': (ws / nv).astype(np.float32),
				'ids': all_sw[0][eom]['ids'],
				'r': all_sw[0][eom]['r'],
				'sdf_ret': sdf_s / nv,
			}

	elapsed = time.time() - t0
	with open(vdir / f'{variant}_{country}_weights.pkl', 'wb') as f:
		pickle.dump(averaged, f)
	print(f'done in {elapsed / 60:.1f} min')
	return averaged, elapsed


# Evaluation

def rank_corr(wts, ref_months):
	keys = sorted(k for k in wts if k in ref_months)
	corrs = []
	for eom in keys:
		if len(wts[eom]['w']) < 10:
			continue
		c, _ = spearmanr(wts[eom]['w'], wts[eom]['r'])
		if not np.isnan(c):
			corrs.append(c)
	return float(np.mean(corrs)) if corrs else 0.0, corrs


def quintile_sim(wts, ref_months):
	keys = sorted(k for k in wts if k in ref_months)
	if not keys:
		return np.array([]), []
	rset               = set(keys[::rebalance_freq])
	ml, li, si, pl, ps, hl = [], set(), set(), set(), set(), []
	for eom in keys:
		m = wts[eom]
		w, r, ids = m['w'], m['r'], m['ids']
		tc = 0.0
		if eom in rset:
			nq = max(1, int(len(w) * 0.20))
			so = np.argsort(w)
			li = set(ids[so[::-1][:nq]].tolist())
			si = set(ids[so[:nq]].tolist())
			to = (len(li - pl) + len(pl - li) + len(si - ps) + len(ps - si)) / max(nq, 1)
			tc = to * tc_bps / 10000.0
			pl, ps = li, si
			hl.append({
				'eom': str(eom),
				'long':  [{'id': i, 'gvkey': id_to_gvkey.get(i, '')} for i in sorted(li)],
				'short': [{'id': i, 'gvkey': id_to_gvkey.get(i, '')} for i in sorted(si)],
			})
		if not li:
			continue
		il = ids.tolist()
		lr = r[np.array([i in li for i in il])]
		sr = r[np.array([i in si for i in il])]
		ml.append((float(lr.mean()) if len(lr) else 0) - (float(sr.mean()) if len(sr) else 0) - tc)
	return np.array(ml), hl


def score_weighted_sim(wts, ref_months):
	ml = []
	for eom in sorted(k for k in wts if k in ref_months):
		w = wts[eom]['w'].astype(np.float64)
		r = wts[eom]['r'].astype(np.float64)
		ww = w - w.mean()
		a = np.abs(ww).sum()
		if a > 1e-10:
			ml.append(float((ww / a) @ r))
	return np.array(ml)


def sdf_returns(wts, ref_months):
	return np.array([
		wts[k]['sdf_ret'] for k in sorted(k for k in wts if k in ref_months) if 'sdf_ret' in wts[k]
	])


def portfolio_metrics(rets, ppy = 12):
	if len(rets) == 0:
		return {}
	tw = float((1 + rets).prod())
	ann_ret = -1.0 if tw <= 0 else float(tw ** (ppy / len(rets)) - 1)
	av = float(rets.std() * np.sqrt(ppy))
	sr = ann_ret / max(av, 1e-8)
	se = float(np.sqrt((1 + 0.5 * sr ** 2) / len(rets)))
	pk = np.maximum.accumulate(np.cumprod(1 + rets))
	dd = float(((pk - np.cumprod(1 + rets)) / pk).max()) if len(pk) else 0
	return {'ann_ret': ann_ret, 'ann_vol': av, 'sharpe': sr, 'se_sharpe': se, 'max_dd': dd, 'n_months': len(rets)}


def evaluate_variant(wts, vname):
	vdir = results_dir / vname
	print(f'\n  {vname}')
	vc, _ = rank_corr(wts, val_months)
	tc_val, tc_monthly = rank_corr(wts, test_months)
	print(f'rank corr val {vc:.4f}   test {tc_val:.4f}')

	qr, holdings = quintile_sim(wts, test_months)
	qm = portfolio_metrics(qr)
	print(f'quintile sharpe {qm.get("sharpe", 0):.4f} (se {qm.get("se_sharpe", 0):.4f})  ret {qm.get("ann_ret", 0) * 100:.2f}%')

	swr = score_weighted_sim(wts, test_months)
	swm = portfolio_metrics(swr)
	print(f'score wt sharpe {swm.get("sharpe", 0):.4f} (se {swm.get("se_sharpe", 0):.4f})  ret {swm.get("ann_ret", 0) * 100:.2f}%')

	sdfr = sdf_returns(wts, test_months)
	sdfm = portfolio_metrics(sdfr)
	print(f'sdf sharpe {sdfm.get("sharpe", 0):.4f} (se {sdfm.get("se_sharpe", 0):.4f})  ret {sdfm.get("ann_ret", 0) * 100:.2f}%')

	np.save(vdir / f'{vname}_{country}_quintile.npy', qr)
	np.save(vdir / f'{vname}_{country}_scorewt.npy', swr)
	np.save(vdir / f'{vname}_{country}_sdf.npy', sdfr)
	with open(vdir / f'{vname}_{country}_holdings.json', 'w') as f:
		json.dump(holdings, f, indent = 2, default = str)
	with open(vdir / f'{vname}_{country}_rank_corrs.json', 'w') as f:
		json.dump({'val': vc, 'test': tc_val, 'monthly': tc_monthly}, f, default = float)

	return {'variant': vname, 'val_corr': vc, 'test_corr': tc_val, 'quintile': qm, 'score_weighted': swm, 'sdf': sdfm}


# Plots

def save_plots(results):
	vs = list(results.keys())
	lb = [v.replace('_', ' ').title() for v in vs]
	x = np.arange(len(vs))

	fig, axes = plt.subplots(2, 2, figsize = (14, 10))
	fig.suptitle(f'Encoding Comparison: {country}', fontsize = 14)

	axes[0, 0].bar(x - 0.17, [results[v]['val_corr'] for v in vs], 0.34, label = 'Val')
	axes[0, 0].bar(x + 0.17, [results[v]['test_corr'] for v in vs], 0.34, label = 'Test')
	axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(lb, rotation = 25, ha = 'right')
	axes[0, 0].set_title('Rank Correlation'); axes[0, 0].legend(); axes[0, 0].grid(axis = 'y', alpha = 0.3)

	for i, (key, title) in enumerate([('quintile', 'Quintile Sharpe'), ('score_weighted', 'Score Weighted'), ('sdf', 'SDF Sharpe')]):
		ax = axes[(i + 1) // 2, (i + 1) % 2]
		sh = [results[v].get(key, {}).get('sharpe', 0) for v in vs]
		se = [results[v].get(key, {}).get('se_sharpe', 0) for v in vs]
		ax.bar(x, sh, yerr = se, capsize = 3)
		ax.set_xticks(x); ax.set_xticklabels(lb, rotation = 25, ha = 'right')
		ax.set_title(title); ax.grid(axis = 'y', alpha = 0.3)

	plt.tight_layout()
	plt.savefig(results_dir / f'{country}_encoding_comparison.png', dpi = 150, bbox_inches = 'tight')
	plt.close()

	fig, ax = plt.subplots(figsize = (10, 5))
	for v in vs:
		p = results_dir / v / f'{v}_{country}_quintile.npy'
		if p.exists():
			ax.plot(np.cumprod(1 + np.load(p)), label = v.replace('_', ' ').title())
	ax.set_title(f'{country}: Cumulative Wealth (Quintile Long Short)')
	ax.set_xlabel('Month'); ax.set_ylabel('Wealth'); ax.legend(); ax.grid(alpha = 0.3)
	plt.tight_layout()
	plt.savefig(results_dir / f'{country}_cumulative_wealth.png', dpi = 150, bbox_inches = 'tight')
	plt.close()
	print(f'\nPlots saved to {results_dir}')


# Main

def main():
	global train_months, val_months, test_months
	global train_gpu, val_gpu, test_gpu
	global char_cols, d, id_to_gvkey

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'PyTorch {torch.__version__}, device: {device}')
	if torch.cuda.is_available():
		print(f'GPU: {torch.cuda.get_device_name(0)}')

	results_dir.mkdir(parents = True, exist_ok = True)

	train_months, val_months, test_months, char_cols, d, id_to_gvkey = process_raw_data()

	train_gpu = to_gpu(train_months, device)
	val_gpu = to_gpu(val_months, device)
	test_gpu = to_gpu(test_months, device)

	results = {}
	for v in variant_list:
		wts, elapsed = train_variant(v, device)
		results[v]   = evaluate_variant(wts, v)
		results[v]['time_min'] = elapsed / 60

	print(f'\n{"Variant":<18} {"Corr":>6} {"Q Sharpe":>9} {"SW Sharpe":>10} {"Ret":>7}')
	for v, r in sorted(results.items(), key = lambda x: -x[1]['test_corr']):
		q = r.get('quintile', {})
		sw = r.get('score_weighted', {})
		print(f'{v:<18} {r["test_corr"]:6.4f} {q.get("sharpe", 0):9.3f} {sw.get("sharpe", 0):10.3f} {q.get("ann_ret", 0) * 100:6.2f}%')

	with open(results_dir / f'{country}_encoding_comparison.json', 'w') as f:
		json.dump(results, f, indent = 2, default = lambda x: float(x) if hasattr(x, '__float__') else str(x))

	save_plots(results)


if __name__ == '__main__':
	main()
