import gc
import json
import math
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

import optuna

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Dataset & Data Loading

class FirmLevelDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, n_classes):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.n_classes = n_classes

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

        features = df[feature_cols].fillna(0.0).values.astype(np.float32)
        self.features = torch.tensor(features, dtype = torch.float32)

        self.targets = {}
        for t_col in target_cols:
            labels = df[t_col + "_label"].values.astype(np.int64)
            self.targets[t_col] = torch.tensor(labels, dtype = torch.long)

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
    needed = feature_cols + target_cols + ["eom", "permno"]
    df = pd.read_parquet(path, columns = needed)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors = "coerce")
    print(f"Loading {path.name}:")
    return FirmLevelDataset(df, feature_cols, target_cols, n_classes)


# Model Architecture

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, d_model))
        self.biases = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a = math.sqrt(5))
        fan_in = 1
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

class FTTransformerBlock(nn.Module):
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
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights

class FTTransformer(nn.Module):
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
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim = 1)

        attn_maps = []
        for block in self.blocks:
            tokens, attn_w = block(tokens)
            attn_maps.append(attn_w)

        cls_out = self.final_norm(tokens[:, 0, :])
        logits = {h: head(cls_out) for h, head in self.heads.items()}
        return logits, attn_maps

# Optimization

def compute_loss(logits, targets, config):
    weights = {
        "target_3m": config.lambda_3m,
        "target_6m": config.lambda_6m,
        "target_12m": config.lambda_12m,
    }
    total = None
    for h, w in weights.items():
        valid = targets[h] >= 0
        if valid.sum() == 0:
            continue
        loss = F.cross_entropy(logits[h][valid], targets[h][valid])
        if total is None:
            total = w * loss
        else:
            total = total + w * loss
    return total if total is not None else torch.tensor(0.0, device=device)

@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = {h: t.to(device) for h, t in targets.items()}

        with torch.autocast("cuda", enabled = torch.cuda.is_available()):
            logits, _ = model(features)
            loss = compute_loss(logits, targets, config)

        total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


# Optuna Objective Function

def objective(trial, train_ds, val_ds, all_feature_cols, target_cols, n_features, base_cfg):

    # Constraint: d_model must be perfectly divisible by n_heads
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_model_per_head = trial.suggest_categorical("d_model_per_head", [8, 16, 32])
    d_model = n_heads * d_model_per_head 
    
    d_ff_ratio = trial.suggest_categorical("d_ff_ratio", [2, 4])
    d_ff = d_model * d_ff_ratio
    
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Instantiate a trial configuration
    trial_cfg = FTConfig(
        train_path=base_cfg.train_path,
        val_path=base_cfg.val_path,
        test_path=base_cfg.test_path,
        results_dir=base_cfg.results_dir,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=base_cfg.max_epochs,
        patience=base_cfg.patience,
        grad_clip=base_cfg.grad_clip,
        lambda_3m=base_cfg.lambda_3m,
        lambda_6m=base_cfg.lambda_6m,
        lambda_12m=base_cfg.lambda_12m,
        n_classes=base_cfg.n_classes,
        seed=base_cfg.seed
    )

    # Enforce exact seeds per trial
    torch.manual_seed(trial_cfg.seed)
    np.random.seed(trial_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_cfg.seed)

    # Initialize dynamic DataLoaders based on sampled batch_size
    train_loader = DataLoader(train_ds, batch_size=trial_cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=trial_cfg.batch_size, shuffle=False, drop_last=False)

    model = FTTransformer(trial_cfg, n_features).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=trial_cfg.learning_rate, weight_decay=trial_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)
    scaler = torch.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    patience_counter = 0

    # Training Loop
    for epoch in range(1, trial_cfg.max_epochs + 1):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = {h: t.to(device) for h, t in targets.items()}

            optimiser.zero_grad()
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, _ = model(features)
                loss = compute_loss(logits, targets, trial_cfg)

            if isinstance(loss, torch.Tensor):
                scaler.scale(loss).backward()
            else:
                loss.backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), trial_cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()

        # Evaluate performance
        val_metrics = evaluate(model, val_loader, trial_cfg)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)

        # Optuna Pruning: Report intermediate validation loss back to look for early termination
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early Stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= trial_cfg.patience:
                break

    # Clean up GPU memory at the end of every trial
    del model, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss


def main():
    cfg = FTConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Classify columns exactly as notebook specifies
    with open("train_columns.json", "r") as f:
        all_columns = json.load(f)

    miss_flags = [c for c in all_columns if c.endswith("_miss")]
    non_miss = [c for c in all_columns if not c.endswith("_miss")]
    lag12_cols = [c for c in non_miss if c.endswith("_lag12")]
    lag12_bases = [c.replace("_lag12", "") for c in lag12_cols]

    k1_chars = sorted([c for c in lag12_bases if c in non_miss])
    all_chars = sorted([c for c in miss_flags if c.replace("_miss", "") in non_miss]) # fallback safety
    all_chars = sorted([c for c in [m.replace("_miss", "") for m in miss_flags] if c in non_miss])
    k0_chars = sorted([c for c in all_chars if c not in k1_chars])
    lag_suffixes = ["", "_lag12", "_lag24", "_lag36", "_lag48", "_lag60"]

    k0_feature_cols = k0_chars.copy()
    k1_feature_cols = []
    for char in k1_chars:
        for suffix in lag_suffixes:
            k1_feature_cols.append(char + suffix)

    all_feature_cols = k0_feature_cols + k1_feature_cols + miss_flags
    target_cols = ["target_3m", "target_6m", "target_12m"]
    n_features = len(all_feature_cols)

    print("Pre-loading training and validation splits...")
    train_ds = load_split(cfg.train_path, all_feature_cols, target_cols, cfg.n_classes)
    val_ds = load_split(cfg.val_path, all_feature_cols, target_cols, cfg.n_classes)

    # Create and configure Optuna Study
    print("\nInitializing Optuna Study...")
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Execute Optimization Loop
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, all_feature_cols, target_cols, n_features, cfg),
        n_trials=20,  # Set target number of parameter combinations to evaluate
        timeout=None
    )

    # Output and Save Results
    print("\nHyperparameter Tuning Complete")
    print(f"Best Validation Loss Achieved: {study.best_value:.4f}")
    print("Best Configuration Parameters:")
    
    best_params = study.best_params
    # Reconstruct explicit d_model and d_ff parameters from the sample factors
    best_params["d_model"] = best_params["n_heads"] * best_params["d_model_per_head"]
    best_params["d_ff"] = best_params["d_model"] * best_params["d_ff_ratio"]

    for param_name, value in best_params.items():
        print(f"  {param_name}: {value}")

    # Serialize best setup directly to configuration directory
    out_path = cfg.results_dir / "ft_transformer_best_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"\nOptimal configuration saved safely to {out_path}")

if __name__ == "__main__":
    main()