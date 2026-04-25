"""Standalone training script mirroring xrd_generator.ipynb.

Used to iterate on model architecture. Selects the architecture variant via the
XRD_VARIANT environment variable so we don't have to maintain many files.
"""
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = int(os.environ.get("XRD_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cascade_pairs", "cascade_pairs.h5")

with h5py.File(H5_PATH, "r") as f:
    n_samples = int(f.attrs["n_samples"])
    xrd_after = f["xrd_after"][:n_samples]
    material_str = [f["material"][i].decode() for i in range(n_samples)]
    structure_str = [f["structure"][i].decode() for i in range(n_samples)]
    lattice_param = f["lattice_param"][:n_samples]
    pka_energy = f["pka_energy"][:n_samples]
    temperature = f["temperature"][:n_samples]

MATERIAL_TO_IDX = {"Cu": 0, "Fe": 1, "Ni": 2, "Pd": 3, "Pt": 4, "W": 5}
STRUCTURE_TO_IDX = {"bcc": 0, "fcc": 1}
material_idx = np.array([MATERIAL_TO_IDX[m] for m in material_str])
structure_idx = np.array([STRUCTURE_TO_IDX[s] for s in structure_str])

continuous = np.stack([lattice_param, pka_energy, temperature], axis=1)
cont_mean = continuous.mean(axis=0)
cont_std = continuous.std(axis=0)
continuous_norm = (continuous - cont_mean) / cont_std

indices = np.arange(n_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=material_idx, random_state=SEED)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=material_idx[test_idx], random_state=SEED)


def make_loader(idx, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(material_idx[idx], dtype=torch.long),
            torch.tensor(structure_idx[idx], dtype=torch.long),
            torch.tensor(continuous_norm[idx], dtype=torch.float32),
            torch.tensor(xrd_after[idx], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx, shuffle=False)
test_loader = make_loader(test_idx, shuffle=False)

# Also build a train-eval loader (no shuffle) so we can score the final train loss
# in eval mode with the same semantics as val — this is the fair comparison.
train_eval_loader = make_loader(train_idx, shuffle=False)

VARIANT = os.environ.get("XRD_VARIANT", "baseline")
WEIGHT_DECAY = float(os.environ.get("XRD_WD", "0"))
LR = float(os.environ.get("XRD_LR", "1e-3"))
MAX_EPOCHS = int(os.environ.get("XRD_EPOCHS", "500"))
PATIENCE = int(os.environ.get("XRD_PATIENCE", "30"))


def make_norm(variant, num_channels):
    """Return the normalization layer per variant."""
    if variant == "baseline":
        return nn.BatchNorm1d(num_channels)
    if variant == "no_norm":
        return nn.Identity()
    if variant == "groupnorm":
        # Pick groups that divide the channel count
        g = 1
        for cand in (8, 4, 2, 1):
            if num_channels % cand == 0:
                g = cand
                break
        return nn.GroupNorm(g, num_channels)
    if variant == "layernorm":
        # LayerNorm over (channels, length) — but length varies, so normalize channel dim via GroupNorm(1, C)
        return nn.GroupNorm(1, num_channels)
    raise ValueError(variant)


class XRDGenerator(nn.Module):
    def __init__(self, variant="baseline", n_materials=6, n_structures=2, n_continuous=3,
                 mat_emb_dim=16, struct_emb_dim=8, init_channels=32, init_length=32):
        super().__init__()
        self.mat_emb = nn.Embedding(n_materials, mat_emb_dim)
        self.struct_emb = nn.Embedding(n_structures, struct_emb_dim)
        input_dim = mat_emb_dim + struct_emb_dim + n_continuous
        self.init_channels = init_channels
        self.init_length = init_length
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, init_channels * init_length),
        )
        # decoder blocks
        def block(cin, cout, last=False):
            layers = [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
            ]
            if not last:
                layers.append(make_norm(variant, cout))
                layers.append(nn.ReLU())
            return layers

        layers = []
        layers += block(32, 32)
        layers += block(32, 16)
        layers += block(16, 8)
        layers += block(8, 1, last=True)
        self.decoder = nn.Sequential(*layers)

    def forward(self, mat_idx, struct_idx, continuous):
        mat_vec = self.mat_emb(mat_idx)
        struct_vec = self.struct_emb(struct_idx)
        x = torch.cat([mat_vec, struct_vec, continuous], dim=1)
        x = self.stem(x)
        x = x.view(-1, self.init_channels, self.init_length)
        x = self.decoder(x)
        x = F.interpolate(x, size=500, mode="linear", align_corners=False)
        return x.squeeze(1)


def eval_loss(model, loader):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for mat, struct, cont, target in loader:
            mat, struct, cont, target = mat.to(device), struct.to(device), cont.to(device), target.to(device)
            pred = model(mat, struct, cont)
            total += F.mse_loss(pred, target, reduction="sum").item() / target.shape[1]
            n += len(mat)
    return total / n


def main():
    print(f"Variant: {VARIANT}  wd={WEIGHT_DECAY}  lr={LR}")
    model = XRDGenerator(variant=VARIANT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        total = 0.0
        n = 0
        for mat, struct, cont, target in train_loader:
            mat, struct, cont, target = mat.to(device), struct.to(device), cont.to(device), target.to(device)
            pred = model(mat, struct, cont)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(mat)
            n += len(mat)
        train_loss_running = total / n

        val_loss = eval_loss(model, val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or bad == PATIENCE:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:3d}  train_running={train_loss_running:.5f}  val={val_loss:.5f}  lr={lr:.1e}  elapsed={time.time()-t0:.0f}s")

        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    # Load best and report fair train/val/test in eval mode
    model.load_state_dict(best_state)
    model = model.to(device)
    train_eval = eval_loss(model, train_eval_loader)
    val_eval = eval_loss(model, val_loader)
    test_eval = eval_loss(model, test_loader)
    gap = val_eval - train_eval
    print()
    print(f"RESULT variant={VARIANT}")
    print(f"  train_eval = {train_eval:.5f}")
    print(f"  val_eval   = {val_eval:.5f}")
    print(f"  test_eval  = {test_eval:.5f}")
    print(f"  val-train  = {gap:+.5f}  {'(val BETTER — over-regularized?)' if gap < 0 else '(val worse — normal)'}")


if __name__ == "__main__":
    main()
