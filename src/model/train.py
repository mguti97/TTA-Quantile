"""Training loop with early stopping on validation loss (Section 7).

The dataset is small (~150 MB), so we keep it *resident on the GPU* and form
mini-batches by indexing a permutation. This avoids the per-batch host->GPU
transfer + DataLoader overhead that otherwise makes this tiny MLP CPU-bound.
If the GPU lacks free memory for the resident tensors (e.g. shared with other
jobs), we fall back to CPU-resident tensors and stream each mini-batch to the
device -- the transfers are tiny (batch x 6 floats) so this stays GPU-bound.

With ``area_lam > 0`` (the canonical player-model recipe, config.AREA_LAM) the
objective gains the soft reachable-area auxiliary term of model/arealoss.py and
early stopping moves to the composite validation loss (pinball on the full val
split + lam * soft-area on a fixed val-state subset); ``area_lam = 0`` recovers
the pure-pinball recipe used by the ball-pass pipeline.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import (
    AREA_AUX_STATES,
    AREA_HORIZONS,
    AREA_VAL_STATES,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LR,
    MAX_EPOCHS,
    MODEL_PT,
    N_FEATURES,
    PINBALL_TAU,
    SEED,
)
from .arealoss import heading_grid, soft_area, stochastic_aux
from .loss import pinball_loss
from .mlp import TTAModel


@torch.no_grad()
def _eval_loss(model, X, y, batch, device, tau):
    """Mean pinball over X, y (any device), streamed to `device` in chunks."""
    model.eval()
    tot = 0.0
    n = X.size(0)
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        loss = pinball_loss(model(X[sl].to(device)), y[sl].to(device), tau=tau)
        tot += loss.item() * (min(i + batch, n) - i)
    return tot / n


def _fits_resident(device, *arrays) -> bool:
    if device.type != "cuda":
        return True
    need = sum(a.nbytes for a in arrays)
    free, _ = torch.cuda.mem_get_info(device)
    return free > need * 1.25 + (1 << 30)  # data + headroom for activations


def train(X_tr, y_tr, X_va, y_va, device=None, verbose=True, in_dim=None,
          tau=PINBALL_TAU, hidden=None, area_lam=0.0, area_horizons=None,
          area_jitter=False, force_streaming=False, seed=SEED):
    """``seed`` controls ONLY training stochasticity (weight init, batch
    shuffling, and the area-aux val-state subsample below) -- the dataset
    (X_tr/y_tr/X_va/y_va, already split upstream) is unaffected."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {"hidden": hidden} if hidden is not None else {}
    model = TTAModel(in_dim=in_dim or N_FEATURES, **kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # resident tensors -- on the GPU when it has room, else CPU + per-batch copy.
    # The free-memory check only sees the raw tensor bytes; at very large row
    # counts (hundreds of millions, e.g. nba_match) actual peak usage across an
    # epoch's ~10^5 fancy-index batches runs well past that estimate, so
    # `force_streaming` lets a caller that already knows the dataset is too big
    # skip the (mis-)estimate entirely.
    resident = not force_streaming and _fits_resident(device, X_tr, y_tr, X_va, y_va)
    data_dev = device if resident else torch.device("cpu")
    if verbose and not resident:
        print("GPU lacks free memory for resident data -> streaming batches from CPU")
    X_tr_t = torch.from_numpy(X_tr).to(data_dev)
    y_tr_t = torch.from_numpy(y_tr).to(data_dev)
    X_va_t = torch.from_numpy(X_va).to(data_dev)
    y_va_t = torch.from_numpy(y_va).to(data_dev)
    n_tr = X_tr_t.size(0)
    eval_batch = BATCH_SIZE * 16

    # area-aware auxiliary term (canonical player recipe; 0 = pure pinball)
    grid_cols = va_states = None
    if area_lam > 0:
        if area_horizons is None:
            area_horizons = AREA_HORIZONS
        grid_cols = heading_grid(device)
        aux_rng = np.random.default_rng(seed + 7)   # jittered grid/horizons
        rng = np.random.default_rng(seed + 3)
        va_idx = rng.choice(X_va_t.size(0), size=min(AREA_VAL_STATES, X_va_t.size(0)),
                            replace=False)
        va_states = X_va_t[torch.from_numpy(va_idx).to(data_dev)][:, 3:].to(device)
        if verbose:
            print(f"area-aware aux: {grid_cols.size(0)} grid cells, lam {area_lam:g}, "
                  f"horizons {tuple(round(h, 3) for h in area_horizons)}"
                  + ("  [jitter: random grid offset + uniform horizons per step]"
                     if area_jitter else ""))

    best_val = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, device=data_dev)
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = X_tr_t[idx].to(device)
            opt.zero_grad()
            loss = pinball_loss(model(xb), y_tr_t[idx].to(device), tau=tau)
            if area_lam > 0:
                if area_jitter:
                    g_cols, hs = stochastic_aux(aux_rng, device, area_horizons)
                else:
                    g_cols, hs = grid_cols, area_horizons
                loss = loss + area_lam * soft_area(model, g_cols,
                                                   xb[:AREA_AUX_STATES, 3:],
                                                   horizons=hs)
            loss.backward()
            opt.step()

        val_pin = _eval_loss(model, X_va_t, y_va_t, eval_batch, device, tau)
        if area_lam > 0:
            with torch.no_grad():
                model.eval()
                val_area = soft_area(model, grid_cols, va_states,
                                     horizons=area_horizons).item()
            val = val_pin + area_lam * val_area
            extra = f"  val_softarea {val_area:.4f}  composite {val:.5f}"
        else:
            val, extra = val_pin, ""
        history.append(val)
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if verbose:
            print(f"epoch {epoch:3d}  val_pinball {val_pin:.5f}{extra}  "
                  f"best {best_val:.5f}  patience {patience}", flush=True)
        if patience >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, history


def save_model(model, path=MODEL_PT):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
