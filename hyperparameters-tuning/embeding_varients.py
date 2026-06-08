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

	learning_rate: float = 1e-4
	weight_decay: float = 1e-5
	max_epochs: int = 50
	patience: int = 7
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


class CrossSectionalDataset(Dataset):
	def __init__(self, df, k0_cols, k1_cols, miss_cols, target_cols_list, max_firms):
		self.max_firms = max_firms
		self.target_col_names = target_cols_list

		dates = sorted(df["eom"].unique())
		self.monthly_data = []

		n_k1 = len(k1_chars)
		for date in dates:
			group = df[df["eom"] == date]
			if len(group) > max_firms:
				group = group.sample(n=max_firms, random_state=42)

			k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
			k1_raw = group[k1_cols].values.astype(np.float32)
			k1 = torch.tensor(k1_raw.reshape(len(group), n_k1, 6), dtype=torch.float32)
			miss = torch.tensor(group[miss_cols].values, dtype=torch.float32)

			targets = {}
			valid_masks = {}
			for tc in target_cols_list:
				vals = group[tc].values.copy().astype(np.float32)
				valid_mask = ~np.isnan(vals)
				vals[~valid_mask] = 0.0
				targets[tc] = torch.tensor(vals, dtype=torch.float32)
				valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)

			self.monthly_data.append({
				"k0": k0,
				"k1": k1,
				"miss": miss,
				"targets": targets,
				"valid_masks": valid_masks,
				"n_firms": len(group),
			})

		del df
		gc.collect()

	def __len__(self):
		return len(self.monthly_data)

	def __getitem__(self, idx):
		return self.monthly_data[idx]


def load_split(path, k0_cols, k1_cols, miss_cols, target_cols_list, max_firms):
	required = k0_cols + k1_cols + miss_cols + target_cols_list + ["eom"]
	df = pd.read_parquet(path, columns=required)

	for col in k0_cols + k1_cols + miss_cols:
		df[col] = df[col].fillna(0.0)

	return CrossSectionalDataset(df, k0_cols, k1_cols, miss_cols, target_cols_list, max_firms)


print("Loading datasets into global shared memory context...")
shared_train_ds = load_split(cfg.train_path, k0_feature_cols, k1_feature_cols,
	miss_flags, target_cols, cfg.max_firms
)
shared_val_ds = load_split(cfg.val_path, k0_feature_cols, k1_feature_cols,
	miss_flags, target_cols, cfg.max_firms
)
print(f"Train months: {len(shared_train_ds)}, Val months: {len(shared_val_ds)}\n")


# Model Architecture

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
		activations = torch.clamp((x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0)
		return activations

	def forward(self, x):
		bin_act = self._encode_bins(x)
		out = torch.einsum("bnk,nkd->bnd", bin_act, self.feature_weights)
		return out


class PeriodicEncoder(nn.Module):
	def __init__(self, n_features, d_model, num_freq=32):
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
	def __init__(self, n_features, d_model, num_freq=32):
		super().__init__()
		self.num_freq = num_freq
		self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
		self.proj = nn.Linear(num_freq * 2, d_model)

	def forward(self, x):
		x_exp = x.unsqueeze(-1)
		scaled = x_exp * self.omega.unsqueeze(0)
		features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
		out = self.proj(features)
		return out


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
		raise ValueError(f"Unknown encoding variant: {variant}")


class SparseMultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, top_k, dropout=0.1):
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
		topk_vals, _ = scores.topk(k, dim=-1)
		threshold = topk_vals[..., -1:].detach()
		mask = scores < threshold
		scores = scores.masked_fill(mask, float("-inf"))

		attn_weights = F.softmax(scores, dim=-1)
		attn_weights = self.dropout(attn_weights)

		context = torch.matmul(attn_weights, V)
		context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
		out = self.W_o(context).squeeze(0)

		return out, attn_weights.squeeze(0)


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


class PortfolioTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		n_k0 = len(k0_chars)
		n_k1 = len(k1_chars)
		n_miss = len(miss_flags)

		self.k0_encoder = build_encoder(
			config.encoding_variant, n_k0, config.d_model,
			ple_bins=config.ple_num_bins,
			periodic_freq=config.periodic_num_freq
		)
		self.k1_encoder = build_encoder(
			config.encoding_variant, n_k1, config.d_model,
			ple_bins=config.ple_num_bins,
			periodic_freq=config.periodic_num_freq
		)

		self.time2vec = Time2Vec(config.d_model)
		self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
		self.miss_proj = nn.Linear(n_miss, config.d_model)

		self.blocks = nn.ModuleList([
			TransformerBlock(
				config.d_model, config.n_heads, config.d_ff,
				config.top_k_attention, config.dropout
			)
			for _ in range(config.n_layers)
		])

		self.head_3m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.head_6m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
		self.head_12m = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

		self.register_buffer("lag_positions", torch.tensor(lag_positions, dtype=torch.float32))

	def _encode_firm_token(self, k0, k1, miss):
		n_firms = k0.shape[0]

		k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
		k0_token = k0_encoded.sum(dim=1)

		k1_flat = k1.permute(0, 2, 1).reshape(n_firms * 6, -1)
		k1_encoded = self.k1_encoder(k1_flat)
		k1_encoded = k1_encoded.view(n_firms, 6, len(k1_chars), self.config.d_model)

		t2v_all = self.time2vec(self.lag_positions).unsqueeze(0).unsqueeze(2)
		k1_encoded = k1_encoded + t2v_all

		k1_token = k1_encoded.sum(dim=(1, 2))

		miss_token = self.miss_proj(miss)
		return k0_token + k1_token + miss_token

	def forward(self, k0, k1, miss):
		z = self._encode_firm_token(k0, k1, miss)

		all_attn = []
		for block in self.blocks:
			z, attn_w = block(z)
			all_attn.append(attn_w)

		return self.head_3m(z).squeeze(-1), self.head_6m(z).squeeze(-1), self.head_12m(z).squeeze(-1), all_attn



# Loss Function (Huber regression)

def compute_multitask_loss(scores_3m, scores_6m, scores_12m, targets, valid_masks, config):
	"""Multi-horizon Huber regression loss, masked to valid observations."""
	total_loss = torch.tensor(0.0, device=scores_3m.device)

	for horizon, scores, weight in [
		("target_3m", scores_3m, config.lambda_3m),
		("target_6m", scores_6m, config.lambda_6m),
		("target_12m", scores_12m, config.lambda_12m),
	]:
		valid = valid_masks[horizon]
		if valid.sum() > 0:
			loss = F.huber_loss(scores[valid], targets[horizon][valid], delta=1.0)
			total_loss = total_loss + weight * loss

	return total_loss



# Hyperparameter Tuning

def objective(trial, variant, gpu_id):
	torch.set_num_threads(1)
	target_device = torch.device(f"cuda:{gpu_id}")
	torch.cuda.set_device(target_device)

	config = Config()
	config.encoding_variant = variant
	config.max_epochs = 40
	config.patience = 8

	# Architecture
	config.d_model = trial.suggest_categorical("d_model", [64, 96, 128])
	valid_heads = [h for h in [2, 4, 8] if config.d_model % h == 0]
	config.n_heads = trial.suggest_categorical("n_heads", valid_heads)
	config.d_ff = trial.suggest_categorical("d_ff_mult", [2, 4]) * config.d_model
	config.n_layers = trial.suggest_int("n_layers", 1, 3)
	config.dropout = trial.suggest_float("dropout", 0.01, 0.4)
	config.top_k_attention = trial.suggest_categorical("top_k", [10, 20, 50, 100])

	# Optimiser
	config.learning_rate = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
	config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
	config.grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0, log=True)

	# Multi-task horizon weights
	config.lambda_3m = trial.suggest_float("lambda_3m", 0.05, 0.45)
	config.lambda_12m = trial.suggest_float("lambda_12m", 0.05, 0.45)
	config.lambda_6m = 1.0 - config.lambda_3m - config.lambda_12m

	# Encoding-specific hyperparameters
	config.ple_num_bins = trial.suggest_categorical("ple_num_bins", [8, 16, 32])
	config.periodic_num_freq = trial.suggest_categorical("periodic_num_freq", [16, 32, 64])

	# Guard against OOM on large configurations
	if config.d_model >= 128 and config.n_layers >= 3:
		config.learning_rate = min(config.learning_rate, 5e-4)

	print(
		f"Trial {trial.number} on cuda:{gpu_id} | "
		f"d={config.d_model} heads={config.n_heads} layers={config.n_layers} "
		f"drop={config.dropout:.2f} lr={config.learning_rate:.1e}"
	)
	sys.stdout.flush()

	model = PortfolioTransformer(config).to(target_device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
	)
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
				miss = batch["miss"].to(target_device, non_blocking=True)
				targets = {k: v.to(target_device, non_blocking=True) for k, v in batch["targets"].items()}
				valid_masks = {k: v.to(target_device, non_blocking=True) for k, v in batch["valid_masks"].items()}

				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type="cuda"):
					s3, s6, s12, _ = model(k0, k1, miss)
					loss = compute_multitask_loss(s3, s6, s12, targets, valid_masks, config)

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

			# Validation: 6-month rebalancing, quintile (20%) long-short Sharpe
			model.eval()
			val_returns = []

			with torch.no_grad():
				for idx in range(0, len(shared_val_ds), 6):
					batch = shared_val_ds.monthly_data[idx]
					k0 = batch["k0"].to(target_device, non_blocking=True)
					k1 = batch["k1"].to(target_device, non_blocking=True)
					miss = batch["miss"].to(target_device, non_blocking=True)
					returns_6m = batch["targets"]["target_6m"].to(target_device, non_blocking=True)
					valid_6m = batch["valid_masks"]["target_6m"].to(target_device, non_blocking=True)

					_, s6, _, _ = model(k0, k1, miss)

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

	DB_URL = "sqlite:///hpt_optuna.db"

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
			study_name=f"hpt_sharpe_{variant}",
			storage=DB_URL,
			load_if_exists=True
		)
		study.optimize(
			variant_objective,
			n_trials=n_trials,
			callbacks=[trial_callback],
			n_jobs=1
		)

		valid_trials = []
		for t in study.trials:
			if t.value is not None and t.datetime_complete is not None and t.datetime_start is not None:
				if 0.0 <= t.value <= 6.0:
					valid_trials.append({
						"number": t.number,
						"value": t.value,
						"state": str(t.state),
						"params": t.params,
						"duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds(),
					})

		valid_trials.sort(key=lambda x: x["value"], reverse=True)

		best = valid_trials[0] if valid_trials else None
		all_variant_results[variant] = {
			"best_trial": best,
			"top_5": valid_trials[:5],
			"total_trials": len(study.trials),
			"valid_trials": len(valid_trials),
		}

		os.makedirs("/kaggle/working", exist_ok=True)
		with open(f"/kaggle/working/hpt_{variant}.json", "w") as f:
			json.dump(all_variant_results[variant], f, indent=2)

		print()
		if best:
			print(f"Best Configuration for {variant}: Trial {best['number']} | Sharpe: {best['value']:.4f}")
			print(f"Params: {best['params']}")
		else:
			print(f"No valid trials yielding a positive Sharpe ratio found for {variant}")
		print()

	print()
	print("Hyperparameter Tuning (Sharpe Target)\n")
	print(f"{'Variant':<15} {'Best Sharpe':>12} {'Trial':>6} {'Valid/Total':>12}")

	for variant in variants:
		res = all_variant_results[variant]
		if res["best_trial"]:
			sharpe_score = f"{res['best_trial']['value']:.4f}"
			trial_num = f"{res['best_trial']['number']}"
		else:
			sharpe_score = "N/A"
			trial_num = "N/A"
		print(f"{variant:<15} {sharpe_score:>12} {trial_num:>6} {res['valid_trials']:>5}/{res['total_trials']}")

	with open("/kaggle/working/hpt_all_variants.json", "w") as f:
		json.dump(all_variant_results, f, indent=2)

	print()
	print("Combined results saved to /kaggle/working/hpt_all_variants.json")

	del shared_train_ds, shared_val_ds
	gc.collect()
