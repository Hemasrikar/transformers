"""
FT-Transformer Benchmark
"""

import gc
import json
import math
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

device = torch.device("cuda")

@dataclass
class FTConfig:
	train_path: Path = Path("../data/processed/train.parquet")
	val_path: Path = Path("../data/processed/val.parquet")
	test_path: Path = Path("../data/processed/test.parquet")
	results_dir: Path = Path("../results/ft_transformer")

	d_model: int = 64
	n_heads: int = 4
	n_layers: int = 2
	d_ff: int = 128
	dropout: float = 0.1

	batch_size: int = 128
	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	max_epochs: int = 50
	patience: int = 7
	grad_clip: float = 1.0

	lambda_3m: float = 0.2
	lambda_6m: float = 0.5
	lambda_12m: float = 0.3

	n_classes: int = 5
	seed: int = 42

cfg = FTConfig()
cfg.results_dir.mkdir(parents = True, exist_ok = True)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)


with open("train_columns.json", "r") as f:
	all_columns = json.load(f)

miss_flags = [c for c in all_columns if c.endswith("_miss")]
miss_bases = [c.replace("_miss", "") for c in miss_flags]
non_miss = [c for c in all_columns if not c.endswith("_miss")]

lag12_cols = [c for c in non_miss if c.endswith("_lag12")]
lag12_bases = [c.replace("_lag12", "") for c in lag12_cols]

K1_CHARS = sorted([c for c in lag12_bases if c in non_miss])
all_chars = sorted([c for c in miss_bases if c in non_miss])
K0_CHARS = sorted([c for c in all_chars if c not in K1_CHARS])

LAG_SUFFIXES = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]

# Build the flat feature column list for the FT-Transformer
# Every feature is treated as an independent token
k0_feature_cols = K0_CHARS.copy()

k1_feature_cols = []
for char in K1_CHARS:
	for suffix in LAG_SUFFIXES:
		k1_feature_cols.append(char + suffix)

all_feature_cols = k0_feature_cols + k1_feature_cols + miss_flags

target_cols = ["target_3m", "target_6m", "target_12m"]
id_cols = ["permno", "date", "eom"]

n_features = len(all_feature_cols)

print(f"K0 characteristics (current only): {len(K0_CHARS)}")
print(f"K1 characteristics (with lags): {len(K1_CHARS)}")
print(f"K1 feature columns (K1 x 6 lags): {len(k1_feature_cols)}")
print(f"Missingness flags: {len(miss_flags)}")
print(f"Total feature tokens for FT-Transformer: {n_features}")


class FirmLevelDataset(Dataset):
	"""
	Dataset for the FT-Transformer benchmark.
	Each sample is a single firm-month observation with a flat feature vector.
	Targets are quintile labels computed within each monthly cross-section.
	"""

	def __init__(self, df, feature_cols, target_cols, n_classes):
		self.feature_cols = feature_cols
		self.target_cols = target_cols
		self.n_classes = n_classes

		# Compute quintile labels within each month
		df = df.copy()
		for t_col in target_cols:
			label_col = t_col + "_label"
			df[label_col] = -1
			for date, group in df.groupby("eom"):
				valid_mask = group[t_col].notna()
				if valid_mask.sum() < n_classes:
					continue
				vals = group.loc[valid_mask, t_col]
				breakpoints = np.percentile(vals, [20, 40, 60, 80])
				labels = np.digitize(vals, breakpoints, right = False)
				labels = np.clip(labels, 0, n_classes - 1)
				df.loc[vals.index, label_col] = labels

		# Store tensors
		features = df[feature_cols].fillna(0.0).values.astype(np.float32)
		self.features = torch.tensor(features, dtype = torch.float32)

		self.targets = {}
		for t_col in target_cols:
			labels = df[t_col + "_label"].values.astype(np.int64)
			self.targets[t_col] = torch.tensor(labels, dtype = torch.long)

		# Retain raw returns and month indices for portfolio simulation
		self.raw_returns = {}
		for t_col in target_cols:
			raw = df[t_col].fillna(0.0).values.astype(np.float32)
			self.raw_returns[t_col] = torch.tensor(raw, dtype = torch.float32)

		self.eom = df["eom"].values
		self.permno = df["permno"].values

		print(f"  Samples: {len(self.features)}, Features: {self.features.shape[1]}")

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		targets = {t_col: self.targets[t_col][idx] for t_col in self.target_cols}
		return self.features[idx], targets


def load_split(path, feature_cols, target_cols, n_classes):
	"""Load a parquet split and construct the firm-level dataset."""
	needed = feature_cols + target_cols + ["eom", "permno"]
	df = pd.read_parquet(path, columns = needed)
	df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors = "coerce")
	print(f"Loading {path.name}:")
	return FirmLevelDataset(df, feature_cols, target_cols, n_classes)


## FT-Transformer Architecture

class FeatureTokenizer(nn.Module):
	"""
	Per-feature linear projection (Gorishniy et al., 2021).
	Each scalar feature x_k is mapped to e_k = W_k * x_k + b_k in R^d.
	"""

	def __init__(self, n_features, d_model):
		super().__init__()
		self.weights = nn.Parameter(torch.empty(n_features, d_model))
		self.biases = nn.Parameter(torch.empty(n_features, d_model))
		nn.init.kaiming_uniform_(self.weights, a = math.sqrt(5))
		fan_in = 1
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.biases, -bound, bound)

	def forward(self, x):
		# x: (batch, n_features)
		# output: (batch, n_features, d_model)
		return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class FTTransformerBlock(nn.Module):
	"""
	Standard Transformer encoder block with pre-norm LayerNorm.
	Uses the standard feed-forward network rather than GRN
	to maintain architectural distinction from the main portfolio Transformer.
	"""

	def __init__(self, d_model, n_heads, d_ff, dropout):
		super().__init__()
		self.norm1 = nn.LayerNorm(d_model)
		self.attn = nn.MultiheadAttention(
			d_model, n_heads,
			dropout = dropout,
			batch_first = True
		)
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
		# Pre-norm multi-head self-attention
		normed = self.norm1(x)
		attn_out, attn_weights = self.attn(normed, normed, normed)
		x = x + self.drop(attn_out)

		# Pre-norm feed-forward
		x = x + self.ffn(self.norm2(x))
		return x, attn_weights


class FTTransformer(nn.Module):
	"""
	Feature Tokenizer Transformer (Gorishniy et al., 2021).

	Architecture:
		1. Per-feature tokenisation: each scalar to R^d
		2. Prepend learnable [CLS] token
		3. L standard Transformer encoder blocks
		4. [CLS] output to prediction heads per horizon
	"""

	def __init__(self, config, n_features):
		super().__init__()
		d = config.d_model

		self.tokenizer = FeatureTokenizer(n_features, d)
		self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

		self.blocks = nn.ModuleList([
			FTTransformerBlock(d, config.n_heads, config.d_ff, config.dropout)
			for _ in range(config.n_layers)
		])

		self.final_norm = nn.LayerNorm(d)

		self.heads = nn.ModuleDict({
			"target_3m": nn.Linear(d, config.n_classes),
			"target_6m": nn.Linear(d, config.n_classes),
			"target_12m": nn.Linear(d, config.n_classes),
		})

	def forward(self, x):
		batch_size = x.size(0)

		# Tokenise: (batch, n_features, d)
		tokens = self.tokenizer(x)

		# Prepend [CLS]: (batch, n_features + 1, d)
		cls = self.cls_token.expand(batch_size, -1, -1)
		tokens = torch.cat([cls, tokens], dim = 1)

		# Transformer blocks
		attn_maps = []
		for block in self.blocks:
			tokens, attn_w = block(tokens)
			attn_maps.append(attn_w)

		# [CLS] output
		cls_out = self.final_norm(tokens[:, 0, :])

		logits = {h: head(cls_out) for h, head in self.heads.items()}
		return logits, attn_maps


## Training Utilities

def compute_loss(logits, targets, config):
	"""Multi-task cross-entropy loss (Eq. 9 in proposal)."""
	weights = {
		"target_3m": config.lambda_3m,
		"target_6m": config.lambda_6m,
		"target_12m": config.lambda_12m,
	}
	device = logits[next(iter(logits))].device
	total = torch.tensor(0.0, device=device)
	for h, w in weights.items():
		valid = targets[h] >= 0
		if valid.sum() == 0:
			continue
		loss = F.cross_entropy(logits[h][valid], targets[h][valid])
		total = total + w * loss
	return total


def compute_rank_correlation(logits, targets):
	"""Spearman rank correlation between expected quintile score and true labels."""
	correlations = {}
	for h in ["target_3m", "target_6m", "target_12m"]:
		valid = targets[h] >= 0
		if valid.sum() < 10:
			correlations[h] = 0.0
			continue
		probs = F.softmax(logits[h][valid], dim = -1)
		quintile_idx = torch.arange(probs.size(1), device = probs.device, dtype = torch.float32)
		scores = (probs * quintile_idx.unsqueeze(0)).sum(dim = 1)
		true = targets[h][valid].float()
		result = spearmanr(scores.cpu().numpy(), true.cpu().numpy())
		if isinstance(result, tuple):
			corr = float(cast(float, result[0]))
		else:
			corr = float(result.correlation)
		correlations[h] = corr if not np.isnan(corr) else 0.0
	return correlations


def compute_accuracy(logits, targets):
	"""Top-1 quintile classification accuracy."""
	accuracies = {}
	for h in ["target_3m", "target_6m", "target_12m"]:
		valid = targets[h] >= 0
		if valid.sum() == 0:
			accuracies[h] = 0.0
			continue
		preds = logits[h][valid].argmax(dim = 1)
		accuracies[h] = (preds == targets[h][valid]).float().mean().item()
	return accuracies

## Evaluation

@torch.no_grad()
def evaluate(model, dataloader, config):
	"""Evaluate on a full split."""
	model.eval()
	all_logits = {h: [] for h in ["target_3m", "target_6m", "target_12m"]}
	all_targets = {h: [] for h in ["target_3m", "target_6m", "target_12m"]}
	total_loss = 0.0
	n_batches = 0

	for features, targets in dataloader:
		features = features.to(device)
		targets = {h: t.to(device) for h, t in targets.items()}

		with torch.autocast("cuda", enabled = torch.cuda.is_available()):
			logits, _ = model(features)
			loss = compute_loss(logits, targets, config)

		total_loss += loss.item()
		n_batches += 1

		for h in all_logits:
			all_logits[h].append(logits[h].cpu())
			all_targets[h].append(targets[h].cpu())

	all_logits = {h: torch.cat(v) for h, v in all_logits.items()}
	all_targets = {h: torch.cat(v) for h, v in all_targets.items()}

	return {
		"loss": total_loss / max(n_batches, 1),
		"rank_corr": compute_rank_correlation(all_logits, all_targets),
		"accuracy": compute_accuracy(all_logits, all_targets),
	}

## Training Loop

def run_training(config):
	"""Full training run with early stopping."""
	print("Loading data splits...")
	train_ds = load_split(config.train_path, all_feature_cols, target_cols, config.n_classes)
	val_ds = load_split(config.val_path, all_feature_cols, target_cols, config.n_classes)

	train_loader = DataLoader(
		train_ds, batch_size = config.batch_size,
		shuffle = True, drop_last = False
	)
	val_loader = DataLoader(
		val_ds, batch_size = config.batch_size,
		shuffle = False, drop_last = False
	)

	model = FTTransformer(config, n_features).to(device)
	param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"FT-Transformer parameters: {param_count:,}")

	optimiser = torch.optim.AdamW(
		model.parameters(),
		lr = config.learning_rate,
		weight_decay = config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimiser, mode = "min", factor = 0.5, patience = 3
	)
	scaler = torch.cuda.amp.GradScaler(enabled = torch.cuda.is_available())

	best_val_loss = float("inf")
	patience_counter = 0
	history = {"train_loss": [], "val_loss": [], "val_corr_6m": []}

	for epoch in range(1, config.max_epochs + 1):
		model.train()
		epoch_loss = 0.0
		n_batches = 0

		for features, targets in train_loader:
			features = features.to(device)
			targets = {h: t.to(device) for h, t in targets.items()}

			optimiser.zero_grad()
			with torch.autocast("cuda", enabled = torch.cuda.is_available()):
				logits, _ = model(features)
				loss = compute_loss(logits, targets, config)

			scaler.scale(loss).backward()
			scaler.unscale_(optimiser)
			nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
			scaler.step(optimiser)
			scaler.update()

			epoch_loss += loss.item()
			n_batches += 1

		train_loss = epoch_loss / max(n_batches, 1)

		val_metrics = evaluate(model, val_loader, config)
		val_loss = val_metrics["loss"]
		scheduler.step(val_loss)
		current_lr = optimiser.param_groups[0]["lr"]

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["val_corr_6m"].append(val_metrics["rank_corr"]["target_6m"])

		print(
			f"Epoch {epoch:3d} | "
			f"Train Loss: {train_loss:.4f} | "
			f"Val Loss: {val_loss:.4f} | "
			f"Val Corr 6m: {val_metrics['rank_corr']['target_6m']:.4f} | "
			f"LR: {current_lr:.2e}"
		)
		sys.stdout.flush()

		if val_loss < best_val_loss - 1e-6:
			best_val_loss = val_loss
			patience_counter = 0
			torch.save(model.state_dict(), config.results_dir / "ft_transformer_best.pt")
		else:
			patience_counter += 1
			if patience_counter >= config.patience:
				print(f"  Early stopping at epoch {epoch}")
				break

	model.load_state_dict(
		torch.load(config.results_dir / "ft_transformer_best.pt", weights_only = True)
	)

	del train_ds, train_loader, val_ds, val_loader
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return model, history

## Training

model, history = run_training(cfg)

## Test Evaluation

test_ds = load_split(cfg.test_path, all_feature_cols, target_cols, cfg.n_classes)
test_loader = DataLoader(test_ds, batch_size = cfg.batch_size, shuffle = False)
test_metrics = evaluate(model, test_loader, cfg)

print("FT-Transformer Test Results")
print(f"  Test Loss: {test_metrics['loss']:.4f}")
print()
for h in ["target_3m", "target_6m", "target_12m"]:
	print(
		f"{h} | Corr: {test_metrics['rank_corr'][h]:.4f} | Acc: {test_metrics['accuracy'][h]:.4f}"
	)

## Portfolio Simulation

@torch.no_grad()
def portfolio_simulation(model, dataset, config, transaction_cost_bps = 25):
	"""Portfolio simulation using the 6-month forecast head."""
	model.eval()
	tc = transaction_cost_bps / 10000.0

	unique_months = sorted(np.unique(dataset.eom))
	monthly_returns = []
	prev_selected = set()

	for month in unique_months:
		month_mask = dataset.eom == month
		month_idx = np.where(month_mask)[0]

		if len(month_idx) < 10:
			continue

		features = dataset.features[month_idx].to(device)
		raw_ret = dataset.raw_returns["target_6m"][month_idx].numpy()
		permnos = dataset.permno[month_idx]

		# Process in sub-batches if the cross-section is large
		all_scores = []
		for start in range(0, len(month_idx), config.batch_size):
			end = min(start + config.batch_size, len(month_idx))
			batch_feat = features[start:end]
			with torch.autocast("cuda", enabled = torch.cuda.is_available()):
				logits, _ = model(batch_feat)
			probs = F.softmax(logits["target_6m"].cpu(), dim = -1)
			quintile_idx = torch.arange(config.n_classes, dtype = torch.float32)
			scores = (probs * quintile_idx.unsqueeze(0)).sum(dim = 1)
			all_scores.append(scores.numpy())

		scores = np.concatenate(all_scores)

		# Select top quintile
		threshold = np.percentile(scores, 80)
		selected_mask = scores >= threshold
		selected_permnos = set(permnos[selected_mask])

		if selected_mask.sum() == 0:
			continue

		port_return = np.mean(raw_ret[selected_mask])

		# Transaction costs on turnover
		if len(prev_selected) > 0:
			entries = len(selected_permnos - prev_selected)
			exits = len(prev_selected - selected_permnos)
			turnover_frac = (entries + exits) / max(len(selected_permnos), 1)
			port_return = port_return - tc * turnover_frac

		monthly_returns.append(port_return)
		prev_selected = selected_permnos

	return np.array(monthly_returns)


def compute_portfolio_metrics(returns):
	"""Compute standard portfolio performance metrics."""
	if len(returns) == 0:
		return {}

	cum_wealth = np.cumprod(1 + returns)
	cum_return = cum_wealth[-1] - 1
	n_months = len(returns)
	annualised_return = (1 + cum_return) ** (12.0 / n_months) - 1
	annualised_vol = np.std(returns) * np.sqrt(12)
	sharpe = annualised_return / annualised_vol if annualised_vol > 0 else 0.0

	peak = np.maximum.accumulate(cum_wealth)
	drawdown = (peak - cum_wealth) / peak
	max_dd = drawdown.max()

	return {
		"cumulative_return": cum_return,
		"annualised_return": annualised_return,
		"annualised_vol": annualised_vol,
		"sharpe_ratio": sharpe,
		"max_drawdown": max_dd,
	}


port_returns = portfolio_simulation(model, test_ds, cfg)
metrics = compute_portfolio_metrics(port_returns)

print("FT-Transformer Portfolio Performance (Top Quintile, Equal Weight, 25 bps TC)")
for k, v in metrics.items():
	print(f"{k}: {v:.4f}")

## Training Diagnostics

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize = (12, 4))

axes[0].plot(history["train_loss"], label = "Train")
axes[0].plot(history["val_loss"], label = "Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("FT-Transformer Loss Curves")
axes[0].legend()

axes[1].plot(history["val_corr_6m"], color = "green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Rank Correlation")
axes[1].set_title("Validation 6-Month Rank Correlation")

plt.tight_layout()
plt.savefig(cfg.results_dir / "ft_transformer_training.png", dpi = 150)
plt.show()


## Comparison Summary

# Populate with your Stage 1 results
stage1_results = {
	"linear": {"corr_6m": None, "acc_6m": None, "loss": None},
	"per_feature": {"corr_6m": None, "acc_6m": None, "loss": None},
	"ple": {"corr_6m": None, "acc_6m": None, "loss": None},
	"periodic": {"corr_6m": None, "acc_6m": None, "loss": None},
}

ft_result = {
	"corr_6m": test_metrics["rank_corr"]["target_6m"],
	"acc_6m": test_metrics["accuracy"]["target_6m"],
	"loss": test_metrics["loss"],
}

print(f"{'Model':<20} {'Test Loss':>10} {'Corr 6m':>10} {'Acc 6m':>10}")
print("=" * 52)
for variant, res in stage1_results.items():
	if res["loss"] is not None:
		print(f"{variant:<20} {res['loss']:>10.4f} {res['corr_6m']:>10.4f} {res['acc_6m']:>10.4f}")
	else:
		print(f"{variant:<20} {'(fill in)':>10} {'(fill in)':>10} {'(fill in)':>10}")
print(f"{'FT-Transformer':<20} {ft_result['loss']:>10.4f} {ft_result['corr_6m']:>10.4f} {ft_result['acc_6m']:>10.4f}")


## Sanity Check & Save Results

test_model = FTTransformer(cfg, n_features).to(device)
param_count = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

dummy_input = torch.randn(32, n_features, device = device)
logits, attn_maps = test_model(dummy_input)

print(f"FT-Transformer parameter count: {param_count:,}")
print(f"Input shape: {dummy_input.shape}")
print(f"Feature tokens (including CLS): {n_features + 1}")
print()
for h, l in logits.items():
	print(f"  {h} logits: {l.shape}")
print()
for i, a in enumerate(attn_maps):
	print(f"  Layer {i} attention: {a.shape}")

del test_model, dummy_input
gc.collect()
torch.cuda.empty_cache()

results_payload = {
	"test_metrics": {
		"loss": test_metrics["loss"],
		"rank_corr": test_metrics["rank_corr"],
		"accuracy": test_metrics["accuracy"],
	},
	"portfolio_metrics": metrics,
	"history": history,
	"config": {
		"d_model": cfg.d_model,
		"n_heads": cfg.n_heads,
		"n_layers": cfg.n_layers,
		"d_ff": cfg.d_ff,
		"batch_size": cfg.batch_size,
		"learning_rate": cfg.learning_rate,
		"n_features": n_features,
	},
}

class NumpyEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, (np.integer,)):
			return int(o)
		if isinstance(o, (np.floating,)):
			return float(o)
		if isinstance(o, np.ndarray):
			return o.tolist()
		return super().default(o)

with open(cfg.results_dir / "ft_transformer_results.json", "w") as f:
	json.dump(results_payload, f, indent = 2, cls = NumpyEncoder)

print(f"Results saved to {cfg.results_dir / 'ft_transformer_results.json'}")

del model, test_ds, test_loader
gc.collect()
torch.cuda.empty_cache()

