import gc
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_cols = ["target_3m", "target_6m", "target_12m"]


@dataclass
class Config:
    # Processed data and output paths
    results_dir: Path = Path("results")
    val_path: Path = Path("data/processed/val.parquet")
    test_path: Path = Path("data/processed/test.parquet")
    col_metadata_path: Path = Path("data/processed/column_metadata.json")
    country_lookup_path: Path = Path("data/processed/country_lookup.parquet")

    # Transformer architecture dimensions (overwritten per variant at load time)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    top_k_attention: int = 50
    ple_num_bins: int = 16
    periodic_num_freq: int = 32

    # Dual path specific parameters
    n_mlp_layers: int = 2
    lambda_aux: float = 0.3
    min_firms_attention: int = 10

    # Portfolio risk management
    target_vol: float = 0.10
    vol_lookback: int = 6
    max_leverage_long_only: float = 3.0
    max_leverage_long_short: float = 3.0
    min_firms_country: int = 20
    max_position_weight: float = 0.05

    encoding_variant: str = "linear"
    seed: int = 24


cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


# Column setup from saved metadata

with open(cfg.col_metadata_path, "r") as f:
    col_meta = json.load(f)

k0_chars = col_meta["retained_k0"]
k1_chars = col_meta["retained_k1"]

parquet_schema_cols = set(pq.read_schema(cfg.test_path).names)

k0_feature_cols = [c for c in k0_chars if c in parquet_schema_cols]
k1_feature_cols = [c for c in k1_chars if c in parquet_schema_cols]

k0_miss_cols = [f"{c}_miss" for c in k0_chars if f"{c}_miss" in parquet_schema_cols]
k1_miss_cols = [f"{c}_miss" for c in k1_chars if f"{c}_miss" in parquet_schema_cols]

country_lookup_df = pd.read_parquet(cfg.country_lookup_path)
country_lookup_df["eom"] = pd.to_datetime(country_lookup_df["eom"])

country_to_id = col_meta["country_to_id"]
country_codes = col_meta["country_codes"]

print(f"K0 characteristics, {len(k0_chars)}")
print(f"K1 characteristics, {len(k1_chars)}")
print(f"K0 missingness flags, {len(k0_miss_cols)}")
print(f"K1 missingness flags, {len(k1_miss_cols)}")
print(f"Countries, {len(country_codes)}")


# Dataset

class CrossSectionalDataset(Dataset):
    """Stores one tensor batch per calendar month. Each batch contains the
    K0 and K1 characteristic tensors, binary missingness flags, integer
    country identifiers, continuous return targets, valid-observation masks,
    a market capitalisation tensor, the firm identifiers from the raw panel,
    and the end-of-month timestamp for all firms in that month."""

    def __init__(self, df, k0_cols, k1_cols, k0_miss_cols, k1_miss_cols,
                 target_col_list, country_lookup, has_market_cap=False):
        dates = sorted(df["eom"].unique())
        self.monthly_data = []

        df = df.merge(country_lookup, on=["id", "eom"], how="left")
        df["country_id"] = df["country_id"].fillna(-1).astype(np.int16)

        for date in dates:
            group = df[df["eom"] == date]

            k0 = torch.tensor(group[k0_cols].values, dtype=torch.float32)
            k1 = torch.tensor(group[k1_cols].values, dtype=torch.float32)
            k0_m = torch.tensor(group[k0_miss_cols].values, dtype=torch.float32)
            k1_m = torch.tensor(group[k1_miss_cols].values, dtype=torch.float32)
            cids = torch.tensor(group["country_id"].values, dtype=torch.long)
            firm_ids = torch.tensor(group["id"].values, dtype=torch.long)

            if has_market_cap:
                market_cap = torch.tensor(group["me"].values, dtype=torch.float32)
            else:
                market_cap = torch.ones(len(group), dtype=torch.float32)

            targets = {}
            valid_masks = {}
            for tc in target_col_list:
                vals = group[tc].values.copy().astype(np.float32)
                valid_mask = ~np.isnan(vals)
                vals[~valid_mask] = 0.0
                targets[tc] = torch.tensor(vals, dtype=torch.float32)
                valid_masks[tc] = torch.tensor(valid_mask, dtype=torch.bool)

            self.monthly_data.append({
                "k0": k0, "k1": k1,
                "k0_miss": k0_m, "k1_miss": k1_m,
                "country_ids": cids,
                "firm_ids": firm_ids,
                "eom": pd.Timestamp(date),
                "market_cap": market_cap,
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


def load_dataset(path, k0_cols, k1_cols, k0_miss, k1_miss,
                 target_col_list, country_lookup):
    """Read a processed parquet split and construct a CrossSectionalDataset."""
    available = set(pq.read_schema(path).names)
    required = ["id", "eom"] + k0_cols + k1_cols + k0_miss + k1_miss + target_col_list
    has_market_cap = "me" in available
    if has_market_cap:
        required = required + ["me"]
    else:
        print(
            f"'me' column not found in {path}. Country composite "
            f"simulation will use firm count weighting."
        )
    load_cols = [c for c in required if c in available]
    df = pd.read_parquet(path, columns=load_cols)
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if col not in df.columns:
            df[col] = 0.0
    for col in k0_cols + k1_cols + k0_miss + k1_miss:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)
    if has_market_cap and df["me"].isna().any():
        df["me"] = df["me"].fillna(0.0)
    return CrossSectionalDataset(
        df, k0_cols, k1_cols, k0_miss, k1_miss,
        target_col_list, country_lookup, has_market_cap=has_market_cap,
    )


# Architecture

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
        return x.unsqueeze(-1) * self.weights + self.biases


class PerFeatureTokeniser(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_model))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x.unsqueeze(-1) * self.weights + self.biases)


class PiecewiseLinearEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        boundaries = torch.linspace(-0.5, 0.5, num_bins + 1)
        self.register_buffer("boundaries", boundaries)
        self.feature_weights = nn.Parameter(
            torch.randn(n_features, num_bins, d_model) * 0.02
        )

    def _encode_bins(self, x):
        t_lower = self.boundaries[:-1]
        t_upper = self.boundaries[1:]
        x_exp = x.unsqueeze(-1)
        activations = torch.clamp(
            (x_exp - t_lower) / (t_upper - t_lower + 1e-8), 0.0, 1.0
        )
        return activations

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
        sinusoidal = torch.sin(
            x_exp * self.omega.unsqueeze(0) + self.phi.unsqueeze(0)
        )
        return self.proj(sinusoidal)


class FourierEncoder(nn.Module):
    def __init__(self, n_features, d_model, num_freq=32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, num_freq) * 0.1)
        self.proj = nn.Linear(num_freq * 2, d_model)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        scaled = x_exp * self.omega.unsqueeze(0)
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(features)


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
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
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
        x_seq = x.unsqueeze(0)
        q = self.w_q(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(1, n_firms, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        k_eff = min(self.top_k, n_firms)
        topk_vals, _ = scores.topk(k_eff, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        scores = scores.masked_fill(scores < threshold, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(1, n_firms, self.d_model)
        out = self.w_o(context).squeeze(0)
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


class AttentiveAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.miss_penalty = nn.Parameter(torch.tensor(5.0))
        self.scale = math.sqrt(d_model)

    def forward(self, encoded, miss_mask=None):
        scores = (encoded * self.query).sum(dim=-1) / self.scale
        if miss_mask is not None:
            penalty = self.miss_penalty.clamp(min=0.0, max=20.0)
            scores = scores - penalty * miss_mask
        weights = F.softmax(scores, dim=1)
        token = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return token, weights


class FirmScoreHead(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, dropout):
        super().__init__()
        modules = [nn.LayerNorm(d_model)]
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_ff
            modules.extend([nn.Linear(in_dim, d_ff), nn.ELU(), nn.Dropout(dropout)])
        final_in = d_ff if n_layers > 0 else d_model
        modules.append(nn.Linear(final_in, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DualPathTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_k0 = len(k0_chars)
        n_k1 = len(k1_chars)

        self.k0_encoder = build_encoder(
            config.encoding_variant, n_k0, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )
        self.k1_encoder = build_encoder(
            config.encoding_variant, n_k1, config.d_model,
            ple_bins=config.ple_num_bins, periodic_freq=config.periodic_num_freq,
        )

        self.k0_static_emb = nn.Parameter(torch.randn(n_k0, config.d_model) * 0.02)
        self.k1_static_emb = nn.Parameter(torch.randn(n_k1, config.d_model) * 0.02)

        self.k0_agg = AttentiveAggregation(config.d_model)
        self.k1_agg = AttentiveAggregation(config.d_model)

        self.base_head_3m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )
        self.base_head_6m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )
        self.base_head_12m = FirmScoreHead(
            config.d_model, config.d_ff, config.n_mlp_layers, config.dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.top_k_attention, config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.adj_head_3m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )
        self.adj_head_6m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )
        self.adj_head_12m = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1)
        )

        self.min_firms = config.min_firms_attention

    def _encode_firms(self, k0, k1, k0_miss, k1_miss):
        k0_encoded = self.k0_encoder(k0) + self.k0_static_emb.unsqueeze(0)
        k0_token, k0_weights = self.k0_agg(k0_encoded, k0_miss)
        k1_encoded = self.k1_encoder(k1) + self.k1_static_emb.unsqueeze(0)
        k1_token, k1_weights = self.k1_agg(k1_encoded, k1_miss)
        z = k0_token + k1_token
        agg_info = {"k0_weights": k0_weights, "k1_weights": k1_weights}
        return z, agg_info

    def forward(self, k0, k1, k0_miss, k1_miss, country_ids):
        z, agg_info = self._encode_firms(k0, k1, k0_miss, k1_miss)

        base_3m = self.base_head_3m(z)
        base_6m = self.base_head_6m(z)
        base_12m = self.base_head_12m(z)

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
            "scores_3m": base_3m + adj_3m, "scores_6m": base_6m + adj_6m,
            "scores_12m": base_12m + adj_12m, "base_3m": base_3m,
            "base_6m": base_6m, "base_12m": base_12m,
            "attn": all_attn, "agg": agg_info,
        }


# Portfolio simulation

def _capped_softmax_weights(scores, max_weight):
    n = scores.shape[0]
    if n == 0:
        return scores.new_zeros(0)
    if max_weight <= 1.0 / n + 1e-12:
        return scores.new_full((n,), 1.0 / n)
    weights = F.softmax(scores, dim=0)
    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights = torch.where(over, torch.full_like(weights, max_weight), weights)
        residual = ~over
        residual_total = weights[residual].sum()
        if residual_total <= 1e-12:
            break
        weights = torch.where(
            residual, weights * (1.0 + excess / residual_total), weights,
        )
    return weights


def _cap_uniform_weights(n, max_weight):
    if n == 0:
        return torch.zeros(0)
    return torch.full((n,), 1.0 / n)


def _firm_id_turnover(prev_ids, curr_ids):
    prev = set(prev_ids.tolist()) if prev_ids is not None else set()
    curr = set(curr_ids.tolist())
    if not curr:
        return 0.0
    new_in = len(curr - prev)
    exited = len(prev - curr)
    return (new_in + exited) / max(len(curr), 1)


def _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key="scores_6m"):
    return torch.stack(
        [m(k0, k1, k0_miss, k1_miss, cids)[key] for m in models]
    ).mean(dim=0)


def _seed_vol_history(models, val_dataset, config, rebalance_freq, leg_kind):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    returns = []
    with torch.no_grad():
        for idx in range(0, len(val_dataset), rebalance_freq):
            batch = val_dataset[idx]
            k0 = batch["k0"].to(device)
            k1 = batch["k1"].to(device)
            k0_miss = batch["k0_miss"].to(device)
            k1_miss = batch["k1_miss"].to(device)
            cids = batch["country_ids"].to(device)
            scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids)
            n_firms = scores.shape[0]
            n_q = max(int(0.2 * n_firms), 1)
            raw = batch["targets"]["target_6m"]
            valid = batch["valid_masks"]["target_6m"]
            _, long_idx = scores.topk(n_q)
            long_idx_np = long_idx.cpu().numpy()
            long_returns = [raw[fi].item() for fi in long_idx_np if valid[fi]]
            long_ret = (
                sum(long_returns) / max(len(long_returns), 1) if long_returns else 0.0
            )
            if leg_kind == "long_only":
                returns.append(long_ret)
            else:
                _, short_idx = scores.topk(n_q, largest=False)
                short_idx_np = short_idx.cpu().numpy()
                short_returns = [raw[fi].item() for fi in short_idx_np if valid[fi]]
                short_ret = (
                    sum(short_returns) / max(len(short_returns), 1) if short_returns else 0.0
                )
                returns.append(long_ret - short_ret)
    return returns[-config.vol_lookback:] if returns else []


@torch.no_grad()
def portfolio_simulation(models, dataset, config, rebalance_freq=6, tc_bps=25,
                         seed_returns=None, record_holdings=False,
                         score_key="scores_6m", target_key="target_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    raw_returns_hist = list(seed_returns) if seed_returns else []
    prev_firm_ids = None
    holdings = []
    leverage_trace = []

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
        n_firms = scores.shape[0]
        n_quintile = max(int(0.2 * n_firms), 1)
        _, top_idx = scores.topk(n_quintile)
        top_idx_np = top_idx.cpu().numpy()
        top_firm_ids = firm_ids[top_idx_np]

        weights = _cap_uniform_weights(n_quintile, config.max_position_weight)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        leg_return = 0.0
        for i, fi in enumerate(top_idx_np):
            if valid[fi]:
                leg_return += weights[i].item() * raw_returns[fi].item()

        base_turnover = _firm_id_turnover(prev_firm_ids, top_firm_ids)

        if len(raw_returns_hist) >= config.vol_lookback:
            recent = np.array(raw_returns_hist[-config.vol_lookback:])
            realised_vol = recent.std() * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(
                leverage,
                1.0 / config.max_leverage_long_only,
                config.max_leverage_long_only,
            ))
        else:
            leverage = 1.0

        tc = leverage * base_turnover * tc_bps / 10000.0
        portfolio_returns.append(leverage * leg_return - tc)
        raw_returns_hist.append(leg_return)
        prev_firm_ids = top_firm_ids

        if record_holdings:
            rebal_idx = idx // rebalance_freq
            leverage_trace.append({
                "rebalance_index": rebal_idx,
                "eom": eom_ts,
                "portfolio": "long_only",
                "leverage": float(leverage),
            })
            for i, fi in enumerate(top_idx_np):
                holdings.append({
                    "rebalance_index": rebal_idx, "eom": eom_ts,
                    "portfolio": "long_only", "leg": "long",
                    "country_id": int(cids[fi].item()), "id": int(firm_ids[fi].item()),
                    "weight": float(weights[i].item()),
                    "realised_return": (float(raw_returns[fi].item()) if valid[fi] else float("nan")),
                })

    returns_arr = np.array(portfolio_returns)
    if record_holdings:
        return returns_arr, holdings, leverage_trace
    return returns_arr


@torch.no_grad()
def portfolio_simulation_long_short(models, dataset, config, rebalance_freq=6, tc_bps=25,
                                    seed_returns=None, record_holdings=False,
                                    score_key="scores_6m", target_key="target_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    raw_ls_returns = list(seed_returns) if seed_returns else []
    prev_long_ids = None
    prev_short_ids = None
    holdings = []
    leverage_trace = []

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
        n_firms = scores.shape[0]
        n_quintile = max(int(0.2 * n_firms), 1)
        _, long_idx = scores.topk(n_quintile)
        _, short_idx = scores.topk(n_quintile, largest=False)
        long_idx_np = long_idx.cpu().numpy()
        short_idx_np = short_idx.cpu().numpy()
        long_firm_ids = firm_ids[long_idx_np]
        short_firm_ids = firm_ids[short_idx_np]

        long_w = _capped_softmax_weights(scores[long_idx], config.max_position_weight)
        short_w = _capped_softmax_weights(-scores[short_idx], config.max_position_weight)

        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        long_ret = 0.0
        for i, fi in enumerate(long_idx_np):
            if valid[fi]:
                long_ret += long_w[i].item() * raw_returns[fi].item()
        short_ret = 0.0
        for i, fi in enumerate(short_idx_np):
            if valid[fi]:
                short_ret += short_w[i].item() * raw_returns[fi].item()
        ls_ret = long_ret - short_ret

        lt = _firm_id_turnover(prev_long_ids, long_firm_ids)
        st = _firm_id_turnover(prev_short_ids, short_firm_ids)
        base_turnover = lt + st

        if len(raw_ls_returns) >= config.vol_lookback:
            recent = np.array(raw_ls_returns[-config.vol_lookback:])
            realised_vol = recent.std() * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(
                leverage, 1.0 / config.max_leverage_long_short,
                config.max_leverage_long_short,
            ))
        else:
            leverage = 1.0

        tc = leverage * base_turnover * tc_bps / 10000.0
        portfolio_returns.append(leverage * ls_ret - tc)
        raw_ls_returns.append(ls_ret)
        prev_long_ids = long_firm_ids
        prev_short_ids = short_firm_ids

        if record_holdings:
            rebal_idx = idx // rebalance_freq
            leverage_trace.append({
                "rebalance_index": rebal_idx,
                "eom": eom_ts,
                "portfolio": "long_short",
                "leverage": float(leverage),
            })
            for i, fi in enumerate(long_idx_np):
                holdings.append({
                    "rebalance_index": rebal_idx,
                    "eom": eom_ts,
                    "portfolio": "long_short",
                    "leg": "long",
                    "country_id": int(cids[fi].item()),
                    "id": int(firm_ids[fi].item()),
                    "weight": float(long_w[i].item()),
                    "realised_return": (
                        float(raw_returns[fi].item()) if valid[fi] else float("nan")
                    ),
                })
            for i, fi in enumerate(short_idx_np):
                holdings.append({
                    "rebalance_index": rebal_idx,
                    "eom": eom_ts,
                    "portfolio": "long_short",
                    "leg": "short",
                    "country_id": int(cids[fi].item()),
                    "id": int(firm_ids[fi].item()),
                    "weight": float(-short_w[i].item()),
                    "realised_return": (
                        float(raw_returns[fi].item()) if valid[fi] else float("nan")
                    ),
                })

    returns_arr = np.array(portfolio_returns)
    if record_holdings:
        return returns_arr, holdings, leverage_trace
    return returns_arr


@torch.no_grad()
def portfolio_simulation_country_composite(models, dataset, config, rebalance_freq=6, tc_bps=25, long_short=True, seed_returns=None,
                                           record_holdings=False, record_per_country=False, score_key="scores_6m", target_key="target_6m"):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()

    periods_per_year = 12 / rebalance_freq
    portfolio_returns = []
    raw_composite_returns = list(seed_returns) if seed_returns else []
    prev_long_ids = {}
    prev_short_ids = {}
    prev_top_ids = {}
    holdings = []
    leverage_trace = []
    per_country = {}

    portfolio_label = (
        "country_composite_long_short" if long_short
        else "country_composite_long_only"
    )
    leverage_bound = (
        config.max_leverage_long_short if long_short
        else config.max_leverage_long_only
    )

    for idx in range(0, len(dataset), rebalance_freq):
        batch = dataset[idx]
        k0 = batch["k0"].to(device)
        k1 = batch["k1"].to(device)
        k0_miss = batch["k0_miss"].to(device)
        k1_miss = batch["k1_miss"].to(device)
        cids = batch["country_ids"].to(device)
        firm_ids = batch["firm_ids"]
        eom_ts = batch["eom"]

        scores = _ensemble_score(models, k0, k1, k0_miss, k1_miss, cids, key=score_key)
        raw_returns = batch["targets"][target_key]
        valid = batch["valid_masks"][target_key]
        market_cap = batch["market_cap"]
        country_ids_np = cids.cpu().numpy()
        rebal_idx = idx // rebalance_freq

        country_returns = {}
        country_costs = {}
        country_market_caps = {}

        for cid in np.unique(country_ids_np):
            if cid < 0:
                continue
            idxs = np.where(country_ids_np == cid)[0]
            n_firms_c = len(idxs)
            if n_firms_c < config.min_firms_country:
                continue

            idxs_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
            scores_c = scores[idxs_t]
            n_quintile_c = max(int(0.2 * n_firms_c), 1)

            if long_short:
                _, long_local = scores_c.topk(n_quintile_c)
                _, short_local = scores_c.topk(n_quintile_c, largest=False)
                long_pos = idxs[long_local.cpu().numpy()]
                short_pos = idxs[short_local.cpu().numpy()]
                long_firm_ids_c = firm_ids[long_pos]
                short_firm_ids_c = firm_ids[short_pos]

                lt = _firm_id_turnover(prev_long_ids.get(cid), long_firm_ids_c)
                st = _firm_id_turnover(prev_short_ids.get(cid), short_firm_ids_c)
                turnover_c = lt + st

                long_w = _capped_softmax_weights(scores[long_pos], config.max_position_weight)
                short_w = _capped_softmax_weights(-scores[short_pos], config.max_position_weight)
                long_ret = 0.0
                for i, fi in enumerate(long_pos):
                    if valid[fi]:
                        long_ret += long_w[i].item() * raw_returns[fi].item()
                short_ret = 0.0
                for i, fi in enumerate(short_pos):
                    if valid[fi]:
                        short_ret += short_w[i].item() * raw_returns[fi].item()
                country_returns[int(cid)] = long_ret - short_ret
                prev_long_ids[cid] = long_firm_ids_c
                prev_short_ids[cid] = short_firm_ids_c

                if record_holdings:
                    for i, fi in enumerate(long_pos):
                        holdings.append({
                            "rebalance_index": rebal_idx, "eom": eom_ts,
                            "portfolio": portfolio_label, "leg": "long",
                            "country_id": int(cid), "id": int(firm_ids[fi].item()),
                            "weight": float(long_w[i].item()),
                            "realised_return": (
                                float(raw_returns[fi].item()) if valid[fi] else float("nan")
                            ),
                        })
                    for i, fi in enumerate(short_pos):
                        holdings.append({
                            "rebalance_index": rebal_idx, "eom": eom_ts,
                            "portfolio": portfolio_label, "leg": "short",
                            "country_id": int(cid), "id": int(firm_ids[fi].item()),
                            "weight": float(-short_w[i].item()),
                            "realised_return": (
                                float(raw_returns[fi].item()) if valid[fi] else float("nan")
                            ),
                        })
            else:
                _, top_local = scores_c.topk(n_quintile_c)
                top_pos = idxs[top_local.cpu().numpy()]
                top_firm_ids_c = firm_ids[top_pos]

                turnover_c = _firm_id_turnover(prev_top_ids.get(cid), top_firm_ids_c)

                top_w = _cap_uniform_weights(n_quintile_c, config.max_position_weight)
                top_w = top_w / top_w.sum() if top_w.sum() > 0 else top_w
                top_ret = 0.0
                for i, fi in enumerate(top_pos):
                    if valid[fi]:
                        top_ret += top_w[i].item() * raw_returns[fi].item()
                country_returns[int(cid)] = top_ret
                prev_top_ids[cid] = top_firm_ids_c

                if record_holdings:
                    for i, fi in enumerate(top_pos):
                        holdings.append({
                            "rebalance_index": rebal_idx, "eom": eom_ts,
                            "portfolio": portfolio_label, "leg": "long",
                            "country_id": int(cid), "id": int(firm_ids[fi].item()),
                            "weight": float(top_w[i].item()),
                            "realised_return": (
                                float(raw_returns[fi].item()) if valid[fi] else float("nan")
                            ),
                        })

            country_costs[int(cid)] = turnover_c * tc_bps / 10000.0
            cap = market_cap[idxs].sum().item()
            country_market_caps[int(cid)] = cap if cap > 0 else float(n_firms_c)

        if not country_returns:
            portfolio_returns.append(0.0)
            raw_composite_returns.append(0.0)
            continue

        total_weight = sum(country_market_caps.values())
        composite_ret = 0.0
        composite_cost = 0.0
        for cid_int, ret in country_returns.items():
            w = country_market_caps[cid_int] / total_weight
            composite_ret += w * ret
            composite_cost += w * country_costs[cid_int]

            if record_per_country:
                entry = per_country.setdefault(cid_int, {
                    "rebalance_indices": [], "returns": [], "weights": [],
                })
                entry["rebalance_indices"].append(rebal_idx)
                entry["returns"].append(float(ret))
                entry["weights"].append(float(w))

        if len(raw_composite_returns) >= config.vol_lookback:
            recent = np.array(raw_composite_returns[-config.vol_lookback:])
            realised_vol = recent.std() * np.sqrt(periods_per_year)
            leverage = config.target_vol / max(realised_vol, 1e-6)
            leverage = float(np.clip(
                leverage, 1.0 / leverage_bound, leverage_bound,
            ))
        else:
            leverage = 1.0

        portfolio_returns.append(leverage * composite_ret - leverage * composite_cost)
        raw_composite_returns.append(composite_ret)

        if record_holdings:
            leverage_trace.append({
                "rebalance_index": rebal_idx,
                "eom": eom_ts,
                "portfolio": portfolio_label,
                "leverage": float(leverage),
            })

    returns_arr = np.array(portfolio_returns)
    out = [returns_arr]
    if record_holdings:
        out.append(holdings)
        out.append(leverage_trace)
    if record_per_country:
        out.append(per_country)
    if len(out) == 1:
        return out[0]
    return tuple(out)


# Metrics

def compute_portfolio_metrics(returns, periods_per_year=2):
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {
            "cumulative_return": 0.0, "annualised_return": 0.0,
            "annualised_vol": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "n_rebalances": 0,
        }
    cum_return = (1 + returns).prod() - 1
    annualised_return = (
        (1 + cum_return) ** (periods_per_year / max(len(returns), 1)) - 1
    )
    annualised_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualised_return / max(annualised_vol, 1e-8)
    cum_wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_wealth)
    drawdown = (peak - cum_wealth) / peak
    return {
        "cumulative_return": cum_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown.max(),
        "n_rebalances": len(returns),
    }


def rolling_sharpe(returns, window_periods, periods_per_year=2):
    returns = np.asarray(returns, dtype=float)
    series = []
    for start in range(0, len(returns) - window_periods + 1):
        window = returns[start:start + window_periods]
        window_cum = (1 + window).prod() - 1
        ann_ret = (1 + window_cum) ** (periods_per_year / window_periods) - 1
        ann_vol = window.std() * np.sqrt(periods_per_year)
        series.append(float(ann_ret / max(ann_vol, 1e-8)))
    if not series:
        return {"series": [], "mean": None, "std": None}
    return {
        "series": series,
        "mean": float(np.mean(series)),
        "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
    }


def compute_portfolio_metrics_extended(returns, periods_per_year=2):
    base = compute_portfolio_metrics(returns, periods_per_year)
    window_3y = int(round(3 * periods_per_year))
    window_5y = int(round(5 * periods_per_year))
    returns_arr = np.asarray(returns, dtype=float)
    base["rolling_sharpe_3y"] = rolling_sharpe(returns_arr, window_3y, periods_per_year)
    if len(returns_arr) >= window_5y:
        head = returns_arr[:window_5y]
        head_cum = (1 + head).prod() - 1
        ann_ret = (1 + head_cum) ** (periods_per_year / window_5y) - 1
        ann_vol = head.std() * np.sqrt(periods_per_year)
        base["sharpe_5y"] = float(ann_ret / max(ann_vol, 1e-8))
    else:
        base["sharpe_5y"] = None
    return base


def _to_native(v):
    if isinstance(v, dict):
        return {kk: _to_native(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_native(vv) for vv in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return [_to_native(x) for x in v.tolist()]
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:
            return float(v.item())
        return _to_native(v.detach().cpu().numpy())
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _jsonable_metrics(metrics_dict):
    return _to_native(metrics_dict)


def _build_per_country_block(per_country_dict, country_codes_list,
                              country_to_id_map, periods_per_year=2):
    id_to_code = {v: k for k, v in country_to_id_map.items()}
    out = {}
    for cid_int, entry in per_country_dict.items():
        country_code = id_to_code.get(cid_int, f"id_{cid_int}")
        returns_arr = np.array(entry["returns"], dtype=float)
        weights_arr = np.array(entry["weights"], dtype=float)
        base = compute_portfolio_metrics_extended(returns_arr, periods_per_year)
        contributions = weights_arr * returns_arr
        time_avg_contrib = (
            float(contributions.mean()) if len(contributions) > 0 else 0.0
        )
        cum_contrib = float(np.prod(1 + contributions) - 1) if len(contributions) > 0 else 0.0
        out[country_code] = {
            "country_code": country_code,
            "country_id": int(cid_int),
            "rebalance_indices": [int(r) for r in entry["rebalance_indices"]],
            "returns_6m": [float(r) for r in entry["returns"]],
            "weights": [float(w) for w in entry["weights"]],
            "annualised_return": base["annualised_return"],
            "annualised_vol": base["annualised_vol"],
            "sharpe_ratio": base["sharpe_ratio"],
            "max_drawdown": base["max_drawdown"],
            "cumulative_return": base["cumulative_return"],
            "rolling_sharpe_3y": base["rolling_sharpe_3y"],
            "sharpe_5y": base["sharpe_5y"],
            "time_average_contribution": time_avg_contrib,
            "cumulative_contribution": cum_contrib,
        }
    return out


# Evaluation

all_results = {}
for metrics_path in sorted(cfg.results_dir.glob("metrics_*.json")):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    variant_name = metrics.get("variant") or metrics_path.stem.replace("metrics_", "")
    all_results[variant_name] = metrics

if not all_results:
    raise RuntimeError(f"No metrics files found in {cfg.results_dir}")

test_ds = load_dataset(
    cfg.test_path, k0_feature_cols, k1_feature_cols,
    k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df,
)
val_ds = load_dataset(
    cfg.val_path, k0_feature_cols, k1_feature_cols,
    k0_miss_cols, k1_miss_cols, target_cols, country_lookup_df,
)

horizons = [
    ("scores_3m", "target_3m", "3m"),
    ("scores_6m", "target_6m", "6m"),
    ("scores_12m", "target_12m", "12m"),
]

variant_test_summary = {}
variant_val_summary = {}
variant_models = {}

for variant_name in all_results:
    cfg.encoding_variant = variant_name

    # restore the architecture hyperparameters that were active when this
    # variant's checkpoint was saved. each variant may have a different
    # n_layers, d_model, d_ff, or encoder-specific field from its own
    # optuna study, and instantiating DualPathTransformer(cfg) with stale
    # values causes a state_dict mismatch on blocks.1.* etc.
    stored_cfg = all_results[variant_name].get("config", {})
    for field in (
        "d_model", "n_heads", "n_layers", "d_ff", "dropout",
        "top_k_attention", "n_mlp_layers", "periodic_num_freq", "ple_num_bins",
    ):
        if field in stored_cfg:
            setattr(cfg, field, stored_cfg[field])

    variant_model = DualPathTransformer(cfg).to(device)
    weights_path = cfg.results_dir / f"weights_{variant_name}.safetensors"
    variant_model.load_state_dict(safetensors_load(str(weights_path)))
    variant_models[variant_name] = variant_model

    val_lo_seed = _seed_vol_history(variant_model, val_ds, cfg, 6, "long_only")
    val_ls_seed = _seed_vol_history(variant_model, val_ds, cfg, 6, "long_short")

    val_ls_returns = portfolio_simulation_long_short(variant_model, val_ds, cfg)
    val_ls_metrics = compute_portfolio_metrics_extended(val_ls_returns)
    variant_val_summary[variant_name] = {
        "rank_corr_6m": all_results[variant_name]
            .get("val_metrics", {})
            .get("rank_corr", {})
            .get("target_6m", float("nan")),
        "sharpe_ls": val_ls_metrics["sharpe_ratio"],
    }

    portfolio_block = {}
    per_country_blocks = {}
    holdings_records = []
    leverage_records = []

    for score_key, target_key, horizon_label in horizons:
        lo_returns = portfolio_simulation(
            variant_model, test_ds, cfg, seed_returns=val_lo_seed,
            record_holdings=(horizon_label == "6m"),
            score_key=score_key, target_key=target_key,
        )
        if isinstance(lo_returns, tuple):
            lo_returns, lo_holdings, lo_leverage = lo_returns
        else:
            lo_holdings, lo_leverage = [], []

        ls_returns = portfolio_simulation_long_short(
            variant_model, test_ds, cfg, seed_returns=val_ls_seed,
            record_holdings=(horizon_label == "6m"),
            score_key=score_key, target_key=target_key,
        )
        if isinstance(ls_returns, tuple):
            ls_returns, ls_holdings, ls_leverage = ls_returns
        else:
            ls_holdings, ls_leverage = [], []

        cc_lo_out = portfolio_simulation_country_composite(
            variant_model, test_ds, cfg, long_short=False,
            seed_returns=val_lo_seed,
            record_holdings=(horizon_label == "6m"),
            record_per_country=(horizon_label == "6m"),
            score_key=score_key, target_key=target_key,
        )
        if isinstance(cc_lo_out, tuple):
            cc_lo_returns = cc_lo_out[0]
            if horizon_label == "6m":
                cc_lo_holdings = cc_lo_out[1]
                cc_lo_leverage = cc_lo_out[2]
                cc_lo_per_country = cc_lo_out[3]
            else:
                cc_lo_holdings, cc_lo_leverage, cc_lo_per_country = [], [], {}
        else:
            cc_lo_returns = cc_lo_out
            cc_lo_holdings, cc_lo_leverage, cc_lo_per_country = [], [], {}

        cc_ls_out = portfolio_simulation_country_composite(
            variant_model, test_ds, cfg, long_short=True,
            seed_returns=val_ls_seed,
            record_holdings=(horizon_label == "6m"),
            record_per_country=(horizon_label == "6m"),
            score_key=score_key, target_key=target_key,
        )
        if isinstance(cc_ls_out, tuple):
            cc_ls_returns = cc_ls_out[0]
            if horizon_label == "6m":
                cc_ls_holdings = cc_ls_out[1]
                cc_ls_leverage = cc_ls_out[2]
                cc_ls_per_country = cc_ls_out[3]
            else:
                cc_ls_holdings, cc_ls_leverage, cc_ls_per_country = [], [], {}
        else:
            cc_ls_returns = cc_ls_out
            cc_ls_holdings, cc_ls_leverage, cc_ls_per_country = [], [], {}

        block = {
            "long_only": compute_portfolio_metrics_extended(lo_returns),
            "long_short": compute_portfolio_metrics_extended(ls_returns),
            "country_composite_long_only":
                compute_portfolio_metrics_extended(cc_lo_returns),
            "country_composite_long_short":
                compute_portfolio_metrics_extended(cc_ls_returns),
        }
        portfolio_block[horizon_label] = block

        if horizon_label == "6m":
            per_country_blocks["country_composite_long_only"] = (
                _build_per_country_block(
                    cc_lo_per_country, country_codes,
                    col_meta["country_to_id"],
                )
            )
            per_country_blocks["country_composite_long_short"] = (
                _build_per_country_block(
                    cc_ls_per_country, country_codes,
                    col_meta["country_to_id"],
                )
            )
            holdings_records.extend(lo_holdings)
            holdings_records.extend(ls_holdings)
            holdings_records.extend(cc_lo_holdings)
            holdings_records.extend(cc_ls_holdings)
            leverage_records.extend(lo_leverage)
            leverage_records.extend(ls_leverage)
            leverage_records.extend(cc_lo_leverage)
            leverage_records.extend(cc_ls_leverage)

    variant_test_summary[variant_name] = {
        "corr_6m": all_results[variant_name]["test_metrics"]
            .get("rank_corr", {}).get("target_6m", float("nan")),
        "sharpe_lo": portfolio_block["6m"]["long_only"]["sharpe_ratio"],
        "sharpe_ls": portfolio_block["6m"]["long_short"]["sharpe_ratio"],
        "vol_ls": portfolio_block["6m"]["long_short"]["annualised_vol"],
    }

    variant_metrics_path = cfg.results_dir / f"metrics_{variant_name}.json"
    with open(variant_metrics_path, "r") as f:
        saved_metrics = json.load(f)
    saved_metrics["portfolio_metrics"] = _jsonable_metrics(portfolio_block["6m"])
    saved_metrics["per_horizon_portfolio_metrics"] = _jsonable_metrics(portfolio_block)
    saved_metrics["per_country"] = _jsonable_metrics(per_country_blocks)
    saved_metrics["validation_portfolio_metrics"] = _jsonable_metrics({
        "long_short": val_ls_metrics,
    })
    with open(variant_metrics_path, "w") as f:
        json.dump(saved_metrics, f, indent=2)

    if holdings_records:
        holdings_df = pd.DataFrame(holdings_records)
        holdings_path = cfg.results_dir / f"holdings_{variant_name}.parquet"
        holdings_df.to_parquet(holdings_path, index=False)
        print(f"Holdings  {holdings_path}  ({len(holdings_df):,} rows)")

    if leverage_records:
        leverage_df = pd.DataFrame(leverage_records)
        leverage_path = cfg.results_dir / f"leverage_{variant_name}.parquet"
        leverage_df.to_parquet(leverage_path, index=False)
        print(f"Leverage  {leverage_path}  ({len(leverage_df):,} rows)")

    print(f"Metrics updated  {variant_metrics_path}")


# Variant selection on validation set metrics

best_variant = max(
    variant_val_summary,
    key=lambda v: variant_val_summary[v]["rank_corr_6m"],
)
best_sharpe_variant = max(
    variant_val_summary,
    key=lambda v: variant_val_summary[v]["sharpe_ls"],
)

print()
print("Variant comparison (test set columns are reported; validation set")
print("columns are the selection criteria):")
print(
    f"{'variant':<12} {'val_corr_6m':>11}  {'val_sharpe_ls':>13}  "
    f"{'test_corr_6m':>12}  {'test_sharpe_lo':>14}  {'test_sharpe_ls':>14}  "
    f"{'test_vol_ls':>11}"
)
for variant_name in all_results:
    vs = variant_val_summary[variant_name]
    ts = variant_test_summary[variant_name]
    marker = ""
    if variant_name == best_variant:
        marker += "  *corr"
    if variant_name == best_sharpe_variant:
        marker += "  *sharpe"
    print(
        f"{variant_name:<12} "
        f"{vs['rank_corr_6m']:>11.4f}  {vs['sharpe_ls']:>13.4f}  "
        f"{ts['corr_6m']:>12.4f}  {ts['sharpe_lo']:>14.4f}  "
        f"{ts['sharpe_ls']:>14.4f}  {ts['vol_ls']:>11.4f}{marker}"
    )
print(f"Best by validation rank correlation, {best_variant}")
print(f"Best by validation long short Sharpe ratio, {best_sharpe_variant}")

# Detailed test set report for best_variant

best_metrics_path = cfg.results_dir / f"metrics_{best_variant}.json"
with open(best_metrics_path, "r") as f:
    best_saved_metrics = json.load(f)
six_month_metrics = best_saved_metrics["portfolio_metrics"]

print(f"Detailed test set portfolio metrics for {best_variant} (best by validation rank correlation):")
print()
for label in ["long_only", "long_short",
              "country_composite_long_only", "country_composite_long_short"]:
    print(f"{label}:")
    m = six_month_metrics[label]
    for k in ("cumulative_return", "annualised_return", "annualised_vol",
              "sharpe_ratio", "max_drawdown", "n_rebalances", "sharpe_5y"):
        v = m.get(k)
        if v is None:
            print(f"  {k}, n/a")
        elif isinstance(v, (int, float)):
            print(f"  {k}, {v:.4f}")
    rs = m.get("rolling_sharpe_3y", {})
    if rs and rs.get("mean") is not None:
        print(f"rolling_sharpe_3y_mean, {rs['mean']:.4f}")
        print(f"rolling_sharpe_3y_std, {rs['std']:.4f}")
    print()


# Cleanup
for variant_name, variant_model in variant_models.items():
    del variant_model
del variant_models, test_ds, val_ds
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
