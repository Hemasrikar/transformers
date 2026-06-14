"""
Dual Path Portfolio Transformer with Attention Weighted Aggregation

"""

import gc
import json
import math
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda")

## Configuration

@dataclass
class Config:
	train_path: Path = Path("data/processed/train.parquet")
	val_path: Path = Path("data/processed/val.parquet")
	test_path: Path = Path("data/processed/test.parquet")
	country_lookup_path: Path = Path("data/processed/country_lookup.parquet")
	col_metadata_path: Path = Path("data/processed/column_metadata.json")
	results_dir: Path = Path("results")

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
	max_epochs: int = 100
	patience: int = 15
	grad_clip: float = 1.0

	lambda_3m: float = 0.2
	lambda_6m: float = 0.5
	lambda_12m: float = 0.3

	encoding_variant: str = "linear"
	max_firms: int = 5000
	seed: int = 24

cfg = Config()
cfg.results_dir.mkdir(parents = True, exist_ok = True)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

## Column Classification

with open(cfg.col_metadata_path, "r") as f:
	col_meta = json.load(f)

with open("../jsons/train_columns.json", "r") as f:
	all_columns = json.load(f)

# Discover which columns are actually present in the training parquet.
# When using a country subset, the column-level missing filter may drop
# characteristics that survive in the full EM panel, causing a mismatch
# between train_columns.json and the parquet schema.
import pyarrow.parquet as pq
_parquet_cols = set(pq.read_schema(cfg.train_path).names)
_missing_from_parquet = [c for c in all_columns if c not in _parquet_cols]
if _missing_from_parquet:
	print(f"Warning: {len(_missing_from_parquet)} columns in train_columns.json "
		  f"are absent from the parquet. Filtering to available columns.")
	print(f"  First 10 dropped: {_missing_from_parquet[:10]}")
all_columns = [c for c in all_columns if c in _parquet_cols]

miss_flags = [c for c in all_columns if c.endswith("_miss")]
miss_bases = [c.replace("_miss", "") for c in miss_flags]
non_miss = [c for c in all_columns if not c.endswith("_miss")]

lag12_cols = [c for c in non_miss if c.endswith("_lag12")]
lag12_bases = [c.replace("_lag12", "") for c in lag12_cols]

K1_CHARS = sorted([c for c in lag12_bases if c in non_miss])
all_chars = sorted([c for c in miss_bases if c in non_miss])
K0_CHARS = sorted([c for c in all_chars if c not in K1_CHARS])

lag_suffixes = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]
LAG_POSITIONS = [0, 12, 24, 36, 48, 60]

k0_feature_cols = K0_CHARS.copy()
# Always include all 6 lag positions per K1 characteristic.
# Lag columns absent from the parquet are zero-filled inside load_split,
# preserving the (n_firms, n_k1, 6) reshape that the dataset constructor requires.
k1_feature_cols = []
for char in K1_CHARS:
	for suffix in lag_suffixes:
		k1_feature_cols.append(char + suffix)

target_cols = ["target_3m", "target_6m", "target_12m"]

k0_miss_cols = [f"{c}_miss" for c in K0_CHARS if f"{c}_miss" in _parquet_cols]
k1_miss_cols = [f"{c}_miss" for c in K1_CHARS if f"{c}_miss" in _parquet_cols]

# Load country lookup from processed output (no raw data dependency)
COUNTRY_LOOKUP = pd.read_parquet(cfg.country_lookup_path)
COUNTRY_LOOKUP["eom"] = pd.to_datetime(COUNTRY_LOOKUP["eom"])

COUNTRY_TO_ID = col_meta["country_to_id"]
COUNTRY_CODES = col_meta["country_codes"]

print(f"K0 characteristics: {len(K0_CHARS)}")
print(f"K1 characteristics: {len(K1_CHARS)} (x6 lags = {len(k1_feature_cols)})")
print(f"K0 miss flags: {len(k0_miss_cols)}")
print(f"K1 miss flags: {len(k1_miss_cols)}")
print(f"Countries: {len(COUNTRY_CODES)}")

## Dataset and Data loading

class CrossSectionalDataset(Dataset):
	def __init__(self, df, k0_cols, k1_cols, k0_miss, k1_miss, target_cols_list,
				 country_lookup, max_firms):
		self.max_firms = max_firms
		dates = sorted(df["eom"].unique())
		self.monthly_data = []

		n_k1 = len(K1_CHARS)
		df = df.merge(country_lookup, on = ["id", "eom"], how = "left")
		df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)

		for date in dates:
			group = df[df["eom"] == date]
			if len(group) > max_firms:
				group = group.sample(n = max_firms, random_state = 42)

			k0 = torch.tensor(group[k0_cols].values, dtype = torch.float32)
			k1_raw = group[k1_cols].values.astype(np.float32)
			k1 = torch.tensor(k1_raw.reshape(len(group), n_k1, 6), dtype = torch.float32)

			k0_m = torch.tensor(group[k0_miss].values, dtype = torch.float32)
			k1_m = torch.tensor(group[k1_miss].values, dtype = torch.float32)
			cids = torch.tensor(group["country_id"].values, dtype = torch.long)

			targets = {}
			valid_masks = {}
			for tc in target_cols_list:
				vals = group[tc].values.copy().astype(np.float32)
				valid_mask = ~np.isnan(vals)
				vals[~valid_mask] = 0.0
				targets[tc] = torch.tensor(vals, dtype = torch.float32)
				valid_masks[tc] = torch.tensor(valid_mask, dtype = torch.bool)

			self.monthly_data.append({
				"k0": k0, "k1": k1,
				"k0_miss": k0_m, "k1_miss": k1_m,
				"country_ids": cids,
				"targets": targets, "valid_masks": valid_masks,
				"n_firms": len(group),
			})

		del df
		gc.collect()

	def __len__(self):
		return len(self.monthly_data)

	def __getitem__(self, idx):
		return self.monthly_data[idx]


def load_split(path, k0_cols, k1_cols, k0_miss, k1_miss, target_cols_list,
			   country_lookup, max_firms):
	required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_cols_list
	available = set(pq.read_schema(path).names)
	missing = [c for c in required if c not in available]
	if missing:
		required = [c for c in required if c in available]
	df = pd.read_parquet(path, columns = required)
	for col in k0_cols + k1_cols + k0_miss + k1_miss:
		if col not in df.columns:
			df[col] = 0.0

	for col in k0_cols + k1_cols + k0_miss + k1_miss:
		df[col] = df[col].fillna(0.0)

	return CrossSectionalDataset(df, k0_cols, k1_cols, k0_miss, k1_miss,
		target_cols_list, country_lookup, max_firms)

## Architecture Components

## Time2Vec Temporal Encoding

class Time2Vec(nn.Module):
	def __init__(self, d_out):
		super().__init__()
		self.d_out = d_out
		self.omega = nn.Parameter(torch.randn(d_out))
		self.phi = nn.Parameter(torch.randn(d_out))

	def forward(self, lag_position):
		lag = lag_position.float().unsqueeze(-1)
		raw = self.omega * lag + self.phi
		out = torch.zeros_like(raw)
		out[..., 0] = raw[..., 0]
		out[..., 1:] = torch.sin(raw[..., 1:])
		return out
	
## Gated REsidual Network

class GRN(nn.Module):
	def __init__(self, d_model, d_ff, dropout = 0.1):
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
		value, gate = gated.chunk(2, dim = -1)
		h = value * torch.sigmoid(gate)
		return self.layer_norm(residual + h)
	
## Feature Encoding Variants

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
	def __init__(self, n_features, d_model, num_bins = 16):
		super().__init__()
		self.num_bins = num_bins
		self.boundaries: torch.Tensor = torch.linspace(-0.5, 0.5, num_bins + 1)
		self.register_buffer("boundaries", self.boundaries)
		self.feature_weights = nn.Parameter(torch.randn(n_features, num_bins, d_model) * 0.02)

	def _encode_bins(self, x):
		t_lower = self.boundaries[:-1]
		t_upper = self.boundaries[1:]
		x_exp = x.unsqueeze(-1)
		activations = torch.clamp((x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0)
		return activations

	def forward(self, x):
		bin_act = self._encode_bins(x)
		out = torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)
		return out


class PeriodicEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq = 32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.phi = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		sinusoidal = torch.sin(x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0))
		out = self.proj(sinusoidal)
		return out
	
class FourierEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq = 32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq * 2, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		scaled = x_exp * self.omega.unsqueeze(0)
		features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim = -1)
		out = self.proj(features)
		return out


def build_encoder(variant, n_features, d_model, ple_bins = 16, periodic_freq = 32):
	if variant == "linear":
		return LinearEncoder(n_features, d_model)
	elif variant == "per_feature":
		return PerFeatureTokeniser(n_features, d_model)
	elif variant == "ple":
		return PiecewiseLinearEncoder(n_features, d_model, num_bins = ple_bins)
	elif variant == "periodic":
		return PeriodicEncoder(n_features, d_model, num_freq = periodic_freq)
	elif variant == "fourier":
		return FourierEncoder(n_features, d_model, num_freq = periodic_freq)
	else:
		raise ValueError(f"Unknown encoding variant: {variant}")
	
## Multi-Head Sparse Attention

class SparseMultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, top_k, dropout = 0.1):
		super().__init__()
		assert d_model % n_heads == 0
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_k = d_model // n_heads
		self.top_k = top_k

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)
		self.W_o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		n_firms = x.shape[0]
		x = x.unsqueeze(0)

		Q = self.W_q(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		K = self.W_k(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
		V = self.W_v(x).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)

		scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

		k = min(self.top_k, n_firms)
		topk_vals, _ = scores.topk(k, dim = -1)
		threshold = topk_vals[..., -1:].detach()
		mask = scores < threshold
		scores = scores.masked_fill(mask, float("-inf"))

		attn_weights = F.softmax(scores, dim = -1)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, V)
		context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
		out = self.W_o(context).squeeze(0)

		return out, attn_weights.squeeze(0)
	
## Transformer Encoder Block

class TransformerBlock(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, top_k, dropout = 0.1):
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
	
## Dual Path Portfolio Transformer

class AttentiveAggregation(nn.Module):
	"""Attention-weighted feature pooling with missingness masking."""
	def __init__(self, d_model):
		super().__init__()
		self.query = nn.Parameter(torch.randn(d_model) * 0.02)
		self.miss_penalty = nn.Parameter(torch.tensor(5.0))
		self.scale = math.sqrt(d_model)

	def forward(self, encoded, miss_mask = None):
		scores = (encoded * self.query).sum(dim = -1) / self.scale
		if miss_mask is not None:
			scores = scores - self.miss_penalty * miss_mask
		weights = F.softmax(scores, dim = 1)
		token = (encoded * weights.unsqueeze(-1)).sum(dim = 1)
		return token, weights


class K1TwoLevelAggregation(nn.Module):
	"""Two-level attention aggregation: first across lags, then across K1 features."""
	def __init__(self, d_model):
		super().__init__()
		self.lag_query = nn.Parameter(torch.randn(d_model) * 0.02)
		self.feat_query = nn.Parameter(torch.randn(d_model) * 0.02)
		self.miss_penalty = nn.Parameter(torch.tensor(5.0))
		self.scale = math.sqrt(d_model)

	def forward(self, k1_encoded, miss_mask = None):
		# k1_encoded: (n_firms, 6, n_k1, d)
		# miss_mask: (n_firms, n_k1)
		lag_scores = (k1_encoded * self.lag_query).sum(dim = -1) / self.scale
		lag_weights = F.softmax(lag_scores, dim = 1)
		h_bar = (k1_encoded * lag_weights.unsqueeze(-1)).sum(dim = 1)

		feat_scores = (h_bar * self.feat_query).sum(dim = -1) / self.scale
		if miss_mask is not None:
			feat_scores = feat_scores - self.miss_penalty * miss_mask
		feat_weights = F.softmax(feat_scores, dim = 1)
		token = (h_bar * feat_weights.unsqueeze(-1)).sum(dim = 1)
		return token, lag_weights, feat_weights


class FirmScoreHead(nn.Module):
	"""Per-firm MLP scoring head with tunable depth."""
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


class DualPathTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		n_k0 = len(K0_CHARS)
		n_k1 = len(K1_CHARS)

		# Feature encoders (shared across both paths)
		self.k0_encoder = build_encoder(
			config.encoding_variant, n_k0, config.d_model,
			ple_bins = config.ple_num_bins, periodic_freq = config.periodic_num_freq
		)
		self.k1_encoder = build_encoder(
			config.encoding_variant, n_k1, config.d_model,
			ple_bins = config.ple_num_bins, periodic_freq = config.periodic_num_freq
		)

		self.time2vec = Time2Vec(config.d_model)
		self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)

		# Attention-weighted aggregation with missingness masking
		self.k0_agg = AttentiveAggregation(config.d_model)
		self.k1_agg = K1TwoLevelAggregation(config.d_model)

		# Path 1: per-firm base score heads
		self.base_head_3m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
		self.base_head_6m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)
		self.base_head_12m = FirmScoreHead(config.d_model, config.d_ff, config.n_mlp_layers, config.dropout)

		# Path 2: cross-sectional attention blocks (shared across countries)
		self.blocks = nn.ModuleList([
			TransformerBlock(config.d_model, config.n_heads, config.d_ff,
				config.top_k_attention, config.dropout)
			for _ in range(config.n_layers)
		])

		# Path 2: adjustment heads (lightweight single linear layer)
		self.adj_head_3m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.adj_head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.adj_head_12m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

		self.register_buffer("lag_positions", torch.tensor(LAG_POSITIONS, dtype = torch.float32))
		self.min_firms = config.min_firms_attention

	def _encode_firms(self, k0, k1, k0_miss, k1_miss):
		n_firms = k0.shape[0]

		# K0 encoding + static embedding + attention aggregation
		k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
		k0_token, k0_weights = self.k0_agg(k0_encoded, k0_miss)

		# K1 encoding + Time2Vec + two-level aggregation
		k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
		k1_encoded = self.k1_encoder(k1_flat)
		k1_encoded = k1_encoded.view(n_firms, 6, len(K1_CHARS), self.config.d_model)
		t2v = self.time2vec(self.lag_positions).unsqueeze(0).unsqueeze(2)
		k1_encoded = k1_encoded + t2v
		k1_token, k1_lag_w, k1_feat_w = self.k1_agg(k1_encoded, k1_miss)

		z = k0_token + k1_token
		agg_info = {"k0_weights": k0_weights, "k1_lag_weights": k1_lag_w, "k1_feat_weights": k1_feat_w}
		return z, agg_info

	def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
		z, agg_info = self._encode_firms(k0, k1, k0_miss, k1_miss)

		# Path 1: per-firm base scores (all firms)
		base_3m = self.base_head_3m(z)
		base_6m = self.base_head_6m(z)
		base_12m = self.base_head_12m(z)

		# Path 2: per-country cross-sectional attention
		adj_3m = torch.zeros_like(base_3m)
		adj_6m = torch.zeros_like(base_6m)
		adj_12m = torch.zeros_like(base_12m)
		all_attn = []

		for cid in country_ids.unique():
			mask = country_ids == cid
			if mask.sum() < self.min_firms:
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
			"base_3m": base_3m, "base_6m": base_6m, "base_12m": base_12m,
			"attn": all_attn,
			"agg": agg_info,
		}
	
## Training Utilities

def compute_dual_path_loss(output, targets, valid_masks, config):
	"""Combined Huber loss on final scores + auxiliary Huber on base scores."""
	main_loss = torch.tensor(0.0, device = output["scores_3m"].device)
	aux_loss = torch.tensor(0.0, device = output["scores_3m"].device)

	for horizon, weight in [("3m", config.lambda_3m), ("6m", config.lambda_6m), ("12m", config.lambda_12m)]:
		target_key = f"target_{horizon}"
		valid = valid_masks[target_key]
		if valid.sum() == 0:
			continue

		t = targets[target_key][valid]
		main_loss = main_loss + weight * F.huber_loss(output[f"scores_{horizon}"][valid], t, delta = 1.0)
		aux_loss = aux_loss + weight * F.huber_loss(output[f"base_{horizon}"][valid], t, delta = 1.0)

	total = main_loss + config.lambda_aux * aux_loss
	return total, main_loss.item(), aux_loss.item()


def compute_rank_correlation(scores, targets, valid_mask):
	"""Spearman rank correlation between predicted scores and continuous returns."""
	valid = valid_mask
	if valid.sum() < 10:
		return 0.0

	pred = scores[valid]
	true = targets[valid]

	def _rank(t):
		order = t.argsort()
		ranks = torch.zeros_like(t)
		ranks[order] = torch.arange(len(t), device = t.device, dtype = torch.float32)
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

## Training and persistance

def train_one_epoch(model, dataset, optimizer, config, scaler):
	model.train()
	epoch_loss = 0.0
	n_months = 0

	indices = np.random.permutation(len(dataset))
	for idx in indices:
		batch = dataset[idx]
		k0 = batch["k0"].to(device, non_blocking = True)
		k1 = batch["k1"].to(device, non_blocking = True)
		k0_miss = batch["k0_miss"].to(device, non_blocking = True)
		k1_miss = batch["k1_miss"].to(device, non_blocking = True)
		cids = batch["country_ids"].to(device, non_blocking = True)
		targets = {k: v.to(device, non_blocking = True) for k, v in batch["targets"].items()}
		valid_masks = {k: v.to(device, non_blocking = True) for k, v in batch["valid_masks"].items()}

		optimizer.zero_grad(set_to_none = True)
		with torch.autocast("cuda"):
			output = model(k0, k1, k0_miss, k1_miss, cids)
			loss, _, _ = compute_dual_path_loss(output, targets, valid_masks, config)

		if loss.requires_grad:
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
			scaler.step(optimizer)
			scaler.update()

		epoch_loss += loss.item()
		n_months += 1

	return epoch_loss / max(n_months, 1)


@torch.no_grad()
def evaluate(model, dataset, config):
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
			h_key = horizon.replace("target_", "scores_")
			total_corr[horizon] += compute_rank_correlation(
				output[h_key], targets[horizon], valid_masks[horizon]
			)

		n_months += 1

	n = max(n_months, 1)
	return {
		"loss": total_loss / n,
		"rank_corr": {k: v / n for k, v in total_corr.items()},
	}


def train_variant(config):
	variant = config.encoding_variant
	print(f"Encoding variant: {variant}")
	print(f"d_model: {config.d_model}, n_heads: {config.n_heads}, n_layers: {config.n_layers}")
	print(f"MLP layers: {config.n_mlp_layers}, lambda_aux: {config.lambda_aux}")
	print(f"Horizon weights: 3m={config.lambda_3m}, 6m={config.lambda_6m}, 12m={config.lambda_12m}")
	print()

	train_ds = load_split(config.train_path, k0_feature_cols, k1_feature_cols,
		k0_miss_cols, k1_miss_cols, target_cols, COUNTRY_LOOKUP, config.max_firms
	)
	val_ds = load_split(config.val_path, k0_feature_cols, k1_feature_cols,
		k0_miss_cols, k1_miss_cols, target_cols, COUNTRY_LOOKUP, config.max_firms
	)
	print(f"Train months: {len(train_ds)}, Val months: {len(val_ds)}")

	model = DualPathTransformer(config).to(device)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {n_params:,}")
	print()

	optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate,
		weight_decay = config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode = "max", factor = 0.5, patience = 10
	)

	best_val_corr = -float("inf")
	patience_counter = 0
	history = {"train_loss": [], "val_loss": [], "val_corr_6m": []}
	weights_path = config.results_dir / f"weights_{variant}.pt"
	scaler = torch.GradScaler("cuda")

	for epoch in range(1, config.max_epochs + 1):
		train_loss = train_one_epoch(model, train_ds, optimizer, config, scaler)
		val_metrics = evaluate(model, val_ds, config)
		val_corr_6m = val_metrics["rank_corr"]["target_6m"]
		scheduler.step(val_corr_6m)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_metrics["loss"])
		history["val_corr_6m"].append(val_corr_6m)

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch:3d} | "
			f"Train Loss:{train_loss:.6f} | "
			f"Val Loss:{val_metrics['loss']:.6f} | "
			f"Val Corr 6m:{val_corr_6m:.4f} | "
			f"LR:{current_lr:.2e}"
		)
		sys.stdout.flush()

		if val_corr_6m > best_val_corr + 1e-5:
			best_val_corr = val_corr_6m
			patience_counter = 0
			torch.save(model.state_dict(), weights_path)
		else:
			patience_counter += 1
			if patience_counter >= config.patience:
				print(f"Early stopping at epoch {epoch}")
				break

	del train_ds, val_ds
	gc.collect()

	model.load_state_dict(torch.load(weights_path, weights_only = True))
	test_ds = load_split(config.test_path, k0_feature_cols, k1_feature_cols,
		k0_miss_cols, k1_miss_cols, target_cols, COUNTRY_LOOKUP, config.max_firms
	)
	test_metrics = evaluate(model, test_ds, config)
	del test_ds

	print(f"\nTest Loss: {test_metrics['loss']:.6f}")
	for h in ["target_3m", "target_6m", "target_12m"]:
		print(f"{h} | Corr:{test_metrics['rank_corr'][h]:.4f}")

	results_path = config.results_dir / f"metrics_{variant}.json"
	results_payload = {
		"variant": variant,
		"n_params": n_params,
		"best_val_corr": best_val_corr,
		"stopped_epoch": len(history["train_loss"]),
		"history": history,
		"test_metrics": test_metrics,
	}
	with open(results_path, "w") as f:
		json.dump(results_payload, f, indent = 2)

	print(f"\nWeights saved to: {weights_path}")
	print(f"Metrics saved to: {results_path}")

	del model
	gc.collect()
	torch.cuda.empty_cache()
	
## Training Varients

cfg.encoding_variant = "linear"
train_variant(cfg)

cfg.encoding_variant = "per_feature"
train_variant(cfg)

cfg.encoding_variant = "ple"
train_variant(cfg)

cfg.encoding_variant = "periodic"
train_variant(cfg)

cfg.encoding_variant = "fourier"
train_variant(cfg)

## Portfolio Simulation

@torch.no_grad()
def portfolio_simulation(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	"""Long-only portfolio: top quintile by 6-month predicted score, equal weighted."""
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
		tc = turnover * transaction_cost_bps / 10000.0

		raw_returns = batch["targets"]["target_6m"]
		valid = batch["valid_masks"]["target_6m"]

		valid_returns = [raw_returns[fi].item() for fi in top_indices.cpu().numpy() if valid[fi]]
		mean_return = sum(valid_returns) / max(len(valid_returns), 1) if valid_returns else 0.0

		portfolio_returns.append(mean_return - tc)
		prev_holdings = top_set

	return np.array(portfolio_returns)


@torch.no_grad()
def portfolio_simulation_long_short(model, dataset, config, rebalance_freq = 6, transaction_cost_bps = 25):
	"""Long-short portfolio: long top quintile, short bottom quintile, score-proportional."""
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
		_, short_idx = scores_6m.topk(n_quintile, largest = False)

		long_set = set(long_idx.cpu().numpy().tolist())
		short_set = set(short_idx.cpu().numpy().tolist())

		lt = len(long_set - prev_long) + len(prev_long - long_set)
		st = len(short_set - prev_short) + len(prev_short - short_set)
		tc = (lt + st) / max(n_quintile, 1) * transaction_cost_bps / 10000.0

		raw_returns = batch["targets"]["target_6m"]
		valid = batch["valid_masks"]["target_6m"]

		long_w = F.softmax(scores_6m[long_idx], dim = 0)
		long_ret = sum(long_w[i].item() * raw_returns[fi].item()
			for i, fi in enumerate(long_idx.cpu().numpy()) if valid[fi])

		short_w = F.softmax(-scores_6m[short_idx], dim = 0)
		short_ret = sum(short_w[i].item() * raw_returns[fi].item()
			for i, fi in enumerate(short_idx.cpu().numpy()) if valid[fi])

		portfolio_returns.append(long_ret - short_ret - tc)
		prev_long = long_set
		prev_short = short_set

	return np.array(portfolio_returns)


def compute_portfolio_metrics(returns, periods_per_year = 2):
	cum_return = (1 + returns).prod() - 1
	annualised_return = (1 + cum_return) ** (periods_per_year / max(len(returns), 1)) - 1
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


# Load saved metrics from disk and choose the best variant based on 6-month rank correlation.
all_results = {}
for metrics_path in sorted(cfg.results_dir.glob("metrics_*.json")):
	with open(metrics_path, "r") as f:
		metrics = json.load(f)
	variant_name = metrics.get("variant") or metrics_path.stem.replace("metrics_", "")
	all_results[variant_name] = metrics

if not all_results:
	raise RuntimeError(f"No metrics files found in {cfg.results_dir}")

best_variant = max(all_results, key = lambda v: all_results[v]["test_metrics"]["rank_corr"]["target_6m"])
print(f"Best variant: {best_variant}")
print()

cfg.encoding_variant = best_variant
best_model = DualPathTransformer(cfg).to(device)
best_model.load_state_dict(torch.load(cfg.results_dir / f"weights_{best_variant}.pt", weights_only = True))

test_ds = load_split(
	cfg.test_path, k0_feature_cols, k1_feature_cols,
	k0_miss_cols, k1_miss_cols, target_cols, COUNTRY_LOOKUP, cfg.max_firms
)

lo_returns = portfolio_simulation(best_model, test_ds, cfg)
ls_returns = portfolio_simulation_long_short(best_model, test_ds, cfg)

print("Long-Only Portfolio:")
for k, v in compute_portfolio_metrics(lo_returns).items():
	print(f"{k}: {v:.4f}")

print()
print("Long-Short Portfolio:")
for k, v in compute_portfolio_metrics(ls_returns).items():
	print(f"{k}: {v:.4f}")

del best_model, test_ds
gc.collect()
torch.cuda.empty_cache()



