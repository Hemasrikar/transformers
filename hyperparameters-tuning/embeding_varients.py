import os
import sys
import gc
import json
import math
import queue
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import cast
import optuna

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


@dataclass
class Config:
	train_path: Path = Path("../data/processed/train.parquet")
	val_path: Path = Path("../data/processed/val.parquet")
	test_path: Path = Path("../data/processed/test.parquet")
	raw_path: Path = Path("../data/Global Factor_EM.parquet")
	results_dir: Path = Path("../results")

	d_model: int = 64
	n_heads: int = 4
	n_layers: int = 2
	d_ff: int = 128
	dropout: float = 0.1

	top_k_attention: int = 50
	time2vec_dim: int = 64
	ple_num_bins: int = 16
	periodic_num_freq: int = 32

	n_mlp_layers: int = 2
	lambda_aux: float = 0.3
	min_firms_attention: int = 10

	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	max_epochs: int = 50
	patience: int = 8
	grad_clip: float = 1.0

	lambda_3m: float = 0.2
	lambda_6m: float = 0.5
	lambda_12m: float = 0.3

	encoding_variant: str = "linear"
	max_firms: int = 5000
	seed: int = 24

cfg = Config()
cfg.results_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

num_gpu = torch.cuda.device_count()
print(f"Using {num_gpu} GPU(s):")
for gpu_id in range(num_gpu):
	print(f"cuda:{gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
print()

gpu_queue = queue.Queue()
for gpu_id in range(num_gpu):
	gpu_queue.put(gpu_id)

def cleanup_cuda(device_obj):
	gc.collect()
	with torch.cuda.device(device_obj):
		torch.cuda.empty_cache()

try:
	with open("../jsons/train_columns.json", "r") as f:
		all_columns = json.load(f)
except FileNotFoundError:
	all_columns = []

miss_flags = [c for c in all_columns if c.endswith("_miss")]
miss_bases = [c.replace("_miss", "") for c in miss_flags]
non_miss = [c for c in all_columns if not c.endswith("_miss")]
lag12_cols = [c for c in non_miss if c.endswith("_lag12")]
lag12_bases = [c.replace("_lag12", "") for c in lag12_cols]

k1_chars = sorted([c for c in lag12_bases if c in non_miss])
all_chars = sorted([c for c in miss_bases if c in non_miss])
k0_chars = sorted([c for c in all_chars if c not in k1_chars])

lag_suffixes = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]
lag_positions = [0, 12, 24, 36, 48, 60]

k0_feature_cols = k0_chars.copy()
k1_feature_cols = []
for char in k1_chars:
	for suffix in lag_suffixes:
		k1_feature_cols.append(char + suffix)

target_cols = ["target_3m", "target_6m", "target_12m"]
k0_miss_cols = [f"{c}_miss" for c in k0_chars]
k1_miss_cols = [f"{c}_miss" for c in k1_chars]

raw_ids = pd.read_parquet(cfg.raw_path, columns=["id", "eom", "excntry"])
raw_ids["eom"] = pd.to_datetime(raw_ids["eom"])
country_codes = sorted(raw_ids["excntry"].unique())
country_to_id = {c: i for i, c in enumerate(country_codes)}
raw_ids["country_id"] = raw_ids["excntry"].map(country_to_id).astype(np.int16)
country_lookup = raw_ids[["id", "eom", "country_id"]].drop_duplicates()
del raw_ids
gc.collect()
print(f"K0: {len(k0_chars)}, K1: {len(k1_chars)} (x6 = {len(k1_feature_cols)}), Countries: {len(country_codes)}\n")


class CrossSectionalDataset(Dataset):
	def __init__(self, df, k0_cols, k1_cols, k0_miss, k1_miss, target_cols_list, country_lookup, max_firms):
		self.max_firms = max_firms
		dates = sorted(df["eom"].unique())
		self.monthly_data = []
		n_k1 = len(k1_chars)
		df = df.merge(country_lookup, on=["id", "eom"], how="left")
		df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)

		for date in dates:
			group = df[df["eom"] == date]
			if len(group) > max_firms:
				group = group.sample(n=max_firms, random_state=42)
			k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
			k1_raw = group[k1_cols].values.astype(np.float32)
			k1 = torch.tensor(k1_raw.reshape(len(group), n_k1, 6), dtype=torch.float32)
			k0_m = torch.tensor(group[k0_miss].values, dtype=torch.float32)
			k1_m = torch.tensor(group[k1_miss].values, dtype=torch.float32)
			cids = torch.tensor(group["country_id"].values, dtype=torch.long)
			targets = {}
			valid_masks = {}
			for tc in target_cols_list:
				vals = group[tc].values.copy().astype(np.float32)
				valid_mask = ~np.isnan(vals)
				vals[~valid_mask] = 0.0
				targets[tc] = torch.tensor(vals, dtype=torch.float32)
				valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)
			self.monthly_data.append({
				"k0": k0, "k1": k1, "k0_miss": k0_m, "k1_miss": k1_m,
				"country_ids": cids, "targets": targets, "valid_masks": valid_masks,
				"n_firms": len(group),
			})
		del df
		gc.collect()

	def __len__(self):
		return len(self.monthly_data)

	def __getitem__(self, idx):
		return self.monthly_data[idx]

def load_split(path, k0_cols, k1_cols, k0_miss, k1_miss, target_cols_list, country_lookup, max_firms):
	required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_cols_list
	df = pd.read_parquet(path, columns=required)
	for col in k0_cols + k1_cols + k0_miss + k1_miss:
		df[col] = df[col].fillna(0.0)
	return CrossSectionalDataset(df, k0_cols, k1_cols, k0_miss, k1_miss, target_cols_list, country_lookup, max_firms)

print("Loading datasets...")
shared_train_ds = load_split(cfg.train_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup, cfg.max_firms)
shared_val_ds = load_split(cfg.val_path, k0_feature_cols, k1_feature_cols, k0_miss_cols, k1_miss_cols, target_cols, country_lookup, cfg.max_firms)
print(f"Train months: {len(shared_train_ds)}, Val months: {len(shared_val_ds)}\n")


class Time2Vec(nn.Module):
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

class LinearEncoder(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))
	def forward(self, x):
		return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

class PerFeatureTokeniser(nn.Module):
	def __init__(self, n_features, d_model):
		super().__init__()
		self.projections = nn.Parameter(torch.randn(n_features, 1, d_model) * 0.02)
		self.biases = nn.Parameter(torch.zeros(n_features, d_model))
	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		proj = self.projections.squeeze(1).unsqueeze(0)
		return x_exp * proj + self.biases.unsqueeze(0)

class PiecewiseLinearEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_bins=16):
		super().__init__()
		self.num_bins = num_bins
		boundaries = torch.linspace(-0.5, 0.5, num_bins + 1)
		self.register_buffer("boundaries", boundaries)
		self.feature_weights = nn.Parameter(torch.randn(n_features, num_bins, d_model) * 0.02)
	def _encode_bins(self, x):
		boundaries = cast(torch.Tensor, self.boundaries)
		t_lower = boundaries[:-1]
		t_upper = boundaries[1:]
		x_exp = x.unsqueeze(-1)
		return torch.clamp((x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0)
	def forward(self, x):
		bin_act = self._encode_bins(x)
		return torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)

class PeriodicEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq=32):
		super().__init__()
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq, d_model)
	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		sinusoidal = torch.sin(x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0))
		return self.proj(sinusoidal)

class FourierEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq=32):
		super().__init__()
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq * 2, d_model)
	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		scaled = x_exp * self.omega.unsqueeze(0)
		return self.proj(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1))

def build_encoder(variant, n_features, d_model, ple_bins=16, periodic_freq=32):
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
		raise ValueError(f"Unknown variant: {variant}")

class SparseMultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, top_k, dropout=0.1):
		super().__init__()
		assert d_model % n_heads == 0
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
		x = x.unsqueeze(0)
		q = self.w_q(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		k = self.w_k(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		v = self.w_v(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
		top_k = min(self.top_k, n_firms)
		topk_vals, _ = scores.topk(top_k, dim=-1)
		threshold = topk_vals[..., -1:].detach()
		mask = scores < threshold
		scores = scores.masked_fill(mask, float("-inf"))
		attn_weights = F.softmax(scores, dim=-1)
		attn_weights = self.dropout(attn_weights)
		context = torch.matmul(attn_weights, v)
		context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
		return self.w_o(context).squeeze(0), attn_weights.squeeze(0)

class TransformerBlock(nn.Module):
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
		return (encoded * weights.unsqueeze(-1)).sum(dim=1), weights

class K1Aggregation(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		self.lag_query = nn.Parameter(torch.randn(d_model) * 0.02)
		self.feat_query = nn.Parameter(torch.randn(d_model) * 0.02)
		self.miss_penalty = nn.Parameter(torch.tensor(5.0))
		self.scale = math.sqrt(d_model)
	def forward(self, k1_encoded, miss_mask=None):
		lag_scores = (k1_encoded * self.lag_query).sum(dim=-1) / self.scale
		lag_weights = F.softmax(lag_scores, dim=1)
		h_bar = (k1_encoded * lag_weights.unsqueeze(-1)).sum(dim=1)
		feat_scores = (h_bar * self.feat_query).sum(dim=-1) / self.scale
		if miss_mask is not None:
			feat_scores = feat_scores - self.miss_penalty * miss_mask
		feat_weights = F.softmax(feat_scores, dim=1)
		return (h_bar * feat_weights.unsqueeze(-1)).sum(dim=1), lag_weights, feat_weights

class FirmScoreHead(nn.Module):
	def __init__(self, d_model, d_ff, n_layers, dropout):
		super().__init__()
		modules: list[nn.Module] = [nn.LayerNorm(d_model)]
		for i in range(n_layers):
			in_dim = d_model if i == 0 else d_ff
			modules.extend([nn.Linear(in_dim, d_ff), nn.ELU(), nn.Dropout(dropout)])
		modules.append(nn.Linear(d_ff if n_layers > 0 else d_model, 1))
		self.net = nn.Sequential(*modules)
	def forward(self, z):
		return self.net(z).squeeze(-1)

class CrossFirmTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		n_k0 = len(k0_chars)
		n_k1 = len(k1_chars)
		self.k0_encoder = build_encoder(config.encoding_variant, n_k0, config.d_model, ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq)
		self.k1_encoder = build_encoder(config.encoding_variant, n_k1, config.d_model, ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq)
		self.time2vec = Time2Vec(config.d_model)
		self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
		self.k0_agg = AttentiveAggregation(config.d_model)
		self.k1_agg = K1Aggregation(config.d_model)
		self.base_head_3m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
		self.base_head_6m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
		self.base_head_12m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
		self.blocks = nn.ModuleList([
			TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.top_k_attention, config.dropout)
			for _ in range(config.n_layers)
		])
		self.adj_head_3m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.adj_head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.adj_head_12m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.register_buffer("lag_positions", torch.tensor(lag_positions, dtype=torch.float32))
		self.min_firms = config.min_firms_attention

	def _encode_firms(self, k0, k1, k0_miss, k1_miss):
		n_firms = k0.shape[0]
		k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
		k0_token, _ = self.k0_agg(k0_encoded, k0_miss)
		k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
		k1_encoded = self.k1_encoder(k1_flat)
		k1_encoded = k1_encoded.view(n_firms, 6, len(k1_chars), self.config.d_model)
		t2v = self.time2vec(self.lag_positions).unsqueeze(0).unsqueeze(2)
		k1_encoded = k1_encoded + t2v
		k1_token, _, _ = self.k1_agg(k1_encoded, k1_miss)
		return k0_token + k1_token

	def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
		z = self._encode_firms(k0, k1, k0_miss, k1_miss)
		base_3m = self.base_head_3m(z)
		base_6m = self.base_head_6m(z)
		base_12m = self.base_head_12m(z)
		adj_3m = torch.zeros_like(base_3m)
		adj_6m = torch.zeros_like(base_6m)
		adj_12m = torch.zeros_like(base_12m)
		for cid in country_ids.unique():
			mask = country_ids == cid
			if mask.sum() < self.min_firms:
				continue
			z_c = z[mask]
			for block in self.blocks:
				z_c, _ = block(z_c)
			adj_3m[mask] = self.adj_head_3m(z_c).squeeze(-1)
			adj_6m[mask] = self.adj_head_6m(z_c).squeeze(-1)
			adj_12m[mask] = self.adj_head_12m(z_c).squeeze(-1)
		return {
			"scores_3m": base_3m + adj_3m, "scores_6m": base_6m + adj_6m, "scores_12m": base_12m + adj_12m,
			"base_3m": base_3m, "base_6m": base_6m, "base_12m": base_12m,
		}


def compute_loss(output, targets, valid_masks, config):
	main_loss = torch.tensor(0.0, device=output["scores_3m"].device)
	aux_loss = torch.tensor(0.0, device=output["scores_3m"].device)
	for horizon, weight in [("3m", config.lambda_3m), ("6m", config.lambda_6m), ("12m", config.lambda_12m)]:
		target_key = f"target_{horizon}"
		valid = valid_masks[target_key]
		if valid.sum() == 0:
			continue
		t = targets[target_key][valid]
		main_loss = main_loss + weight * F.huber_loss(output[f"scores_{horizon}"][valid], t, delta=1.0)
		aux_loss = aux_loss + weight * F.huber_loss(output[f"base_{horizon}"][valid], t, delta=1.0)
	return main_loss + config.lambda_aux * aux_loss


def objective(trial, variant, gpu_id):
	torch.set_num_threads(1)
	target_device = torch.device(f"cuda:{gpu_id}")
	torch.cuda.set_device(target_device)

	config = Config()
	config.encoding_variant = variant
	config.max_epochs = 40
	config.patience = 8

	config.d_model = trial.suggest_categorical("d_model", [64, 96, 128])
	valid_heads = [h for h in [2, 4, 8] if config.d_model % h == 0]
	config.n_heads = trial.suggest_categorical("n_heads", valid_heads)
	config.d_ff = trial.suggest_categorical("d_ff_mult", [2, 4]) * config.d_model
	config.n_layers = trial.suggest_int("n_layers", 1, 3)
	config.dropout = trial.suggest_float("dropout", 0.01, 0.4)
	config.top_k_attention = trial.suggest_categorical("top_k", [10, 20, 50, 100])

	config.n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 3)
	config.lambda_aux = trial.suggest_float("lambda_aux", 0.1, 0.5)

	config.learning_rate = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
	config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
	config.grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0, log=True)

	config.lambda_3m = trial.suggest_float("lambda_3m", 0.05, 0.45)
	config.lambda_12m = trial.suggest_float("lambda_12m", 0.05, 0.45)
	config.lambda_6m = 1.0 - config.lambda_3m - config.lambda_12m

	config.ple_num_bins = trial.suggest_categorical("ple_num_bins", [8, 16, 32])
	config.periodic_num_freq = trial.suggest_categorical("periodic_num_freq", [16, 32, 64])

	if config.d_model >= 128 and config.n_layers >= 3:
		config.learning_rate = min(config.learning_rate, 5e-4)

	print(
		f"Trial {trial.number} on cuda:{gpu_id} | d={config.d_model} heads={config.n_heads} "
		f"layers={config.n_layers} mlp={config.n_mlp_layers} aux={config.lambda_aux:.2f} "
		f"drop={config.dropout:.2f} lr={config.learning_rate:.1e}"
	)
	sys.stdout.flush()

	model = CrossFirmTransformer(config).to(target_device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	scaler = torch.GradScaler()
	best_val_sharpe = -float("inf")
	patience_counter = 0
	train_data = shared_train_ds.monthly_data

	try:
		for epoch in range(1, config.max_epochs + 1):
			model.train()
			nan_detected = False
			for idx in np.random.permutation(len(train_data)):
				batch = train_data[idx]
				k0 = batch["k0"].to(target_device, non_blocking=True)
				k1 = batch["k1"].to(target_device, non_blocking=True)
				k0_m = batch["k0_miss"].to(target_device, non_blocking=True)
				k1_m = batch["k1_miss"].to(target_device, non_blocking=True)
				cids = batch["country_ids"].to(target_device, non_blocking=True)
				targets = {k: v.to(target_device, non_blocking=True) for k, v in batch["targets"].items()}
				valid_masks = {k: v.to(target_device, non_blocking=True) for k, v in batch["valid_masks"].items()}

				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type="cuda"):
					output = model(k0, k1, k0_m, k1_m, cids)
					loss = compute_loss(output, targets, valid_masks, config)

				if torch.isnan(loss) or torch.isinf(loss):
					nan_detected = True
					break
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
				scaler.step(optimizer)
				scaler.update()

			if nan_detected:
				print(f"Trial {trial.number} | Epoch {epoch:2d} | NaN detected, pruning")
				sys.stdout.flush()
				raise optuna.TrialPruned()

			model.eval()
			val_returns = []
			with torch.no_grad():
				for idx in range(0, len(shared_val_ds), 6):
					batch = shared_val_ds.monthly_data[idx]
					k0 = batch["k0"].to(target_device, non_blocking=True)
					k1 = batch["k1"].to(target_device, non_blocking=True)
					k0_m = batch["k0_miss"].to(target_device, non_blocking=True)
					k1_m = batch["k1_miss"].to(target_device, non_blocking=True)
					cids = batch["country_ids"].to(target_device, non_blocking=True)
					returns_6m = batch["targets"]["target_6m"].to(target_device, non_blocking=True)
					valid_6m = batch["valid_masks"]["target_6m"].to(target_device, non_blocking=True)

					output = model(k0, k1, k0_m, k1_m, cids)
					s6 = output["scores_6m"]
					n_assets = s6.shape[0]
					n_quintile = max(1, int(n_assets * 0.20))
					_, long_idx = torch.topk(s6, k=n_quintile, largest=True)
					_, short_idx = torch.topk(s6, k=n_quintile, largest=False)
					long_valid = valid_6m[long_idx]
					long_ret = returns_6m[long_idx][long_valid].mean().item() if long_valid.sum() > 0 else 0.0
					short_valid = valid_6m[short_idx]
					short_ret = returns_6m[short_idx][short_valid].mean().item() if short_valid.sum() > 0 else 0.0
					val_returns.append(long_ret - short_ret)

			returns_np = np.array(val_returns)
			mean_ret = np.mean(returns_np)
			std_ret = np.std(returns_np, ddof=1)
			val_sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(2)

			print(f"Epoch {epoch:2d} | cuda:{gpu_id} | Val Sharpe: {val_sharpe:.4f} | Avg 6m LS Ret: {mean_ret*100:.2f}%")
			sys.stdout.flush()

			if val_sharpe > best_val_sharpe + 1e-4:
				best_val_sharpe = val_sharpe
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= config.patience:
					break
			trial.report(val_sharpe, epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()

		return best_val_sharpe
	finally:
		del model
		cleanup_cuda(target_device)


if __name__ == "__main__":
	variants = ["linear", "per_feature", "ple", "periodic", "fourier"]
	n_trials = 50
	all_variant_results = {}
	DB_URL = "sqlite:///hpt_dual_path.db"

	for variant in variants:
		print(f"Tuning Sharpe Optimisation: {variant.upper()}")
		def variant_objective(trial, v=variant):
			gpu_id = gpu_queue.get()
			try:
				return objective(trial, v, gpu_id)
			finally:
				gpu_queue.put(gpu_id)

		def trial_callback(study, trial):
			status = f"{trial.value:.4f}" if trial.value is not None else "pruned"
			completed_values = [t.value for t in study.trials if t.value is not None]
			best_so_far = max(completed_values, default=0.0)
			print(f"Trial {trial.number:3d} done | Sharpe: {status} | Best Sharpe: {best_so_far:.4f}")
			sys.stdout.flush()

		study = optuna.create_study(
			direction="maximize",
			pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=6),
			study_name=f"hpt_dual_{variant}",
			storage=DB_URL,
			load_if_exists=True
		)
		study.optimize(variant_objective, n_trials=n_trials, callbacks=[trial_callback], n_jobs=1)

		valid_trials = []
		for t in study.trials:
			if t.value is not None and t.datetime_complete is not None and t.datetime_start is not None:
				if 0.0 <= t.value <= 6.0:
					valid_trials.append({
						"number": t.number, "value": t.value, "state": str(t.state),
						"params": t.params,
						"duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds(),
					})
		valid_trials.sort(key=lambda x: x["value"], reverse=True)

		best = valid_trials[0] if valid_trials else None
		all_variant_results[variant] = {
			"best_trial": best, "top_5": valid_trials[:5],
			"total_trials": len(study.trials), "valid_trials": len(valid_trials),
		}
		os.makedirs("/kaggle/working", exist_ok=True)
		with open(f"/kaggle/working/hpt_dual_{variant}.json", "w") as f:
			json.dump(all_variant_results[variant], f, indent=2)
		print()
		if best:
			print(f"Best for {variant}: Trial {best['number']} | Sharpe: {best['value']:.4f}")
			print(f"Params: {best['params']}")
		else:
			print(f"No valid trials for {variant}")
		print()

	print("\nHyperparameter Tuning (Dual Path Sharpe Target)\n")
	print(f"{'Variant':<15} {'Best Sharpe':>12} {'Trial':>6} {'Valid/Total':>12}")
	for variant in variants:
		res = all_variant_results[variant]
		if res["best_trial"]:
			print(f"{variant:<15} {res['best_trial']['value']:>12.4f} {res['best_trial']['number']:>6} {res['valid_trials']:>5}/{res['total_trials']}")
		else:
			print(f"{variant:<15} {'N/A':>12} {'N/A':>6} {res['valid_trials']:>5}/{res['total_trials']}")

	with open("/kaggle/working/hpt_dual_all_variants.json", "w") as f:
		json.dump(all_variant_results, f, indent=2)
	print("\nCombined results saved to /kaggle/working/hpt_dual_all_variants.json")
	del shared_train_ds, shared_val_ds
	gc.collect()
