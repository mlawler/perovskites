"""5-fold stratified CV benchmark for the XRD generator.

Reports mean +/- std of train/val MSE and per-material val MSE across folds,
so we can tell real improvements from data-split luck.
"""
import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

SEED = int(os.environ.get("XRD_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANT = os.environ.get("XRD_VARIANT", "baseline")
LR = float(os.environ.get("XRD_LR", "1e-3"))
WD = float(os.environ.get("XRD_WD", "0"))
MAX_EPOCHS = int(os.environ.get("XRD_EPOCHS", "250"))
PATIENCE = int(os.environ.get("XRD_PATIENCE", "25"))
N_FOLDS = int(os.environ.get("XRD_FOLDS", "5"))
EXTRA_FEATURES = os.environ.get("XRD_EXTRA", "none")  # none|boxdensity

H5_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cascade_pairs", "cascade_pairs.h5")
with h5py.File(H5_PATH, "r") as f:
    n_samples = int(f.attrs["n_samples"])
    xrd_after = f["xrd_after"][:n_samples]
    material_str = [f["material"][i].decode() for i in range(n_samples)]
    structure_str = [f["structure"][i].decode() for i in range(n_samples)]
    lattice_param = f["lattice_param"][:n_samples]
    pka_energy = f["pka_energy"][:n_samples]
    temperature = f["temperature"][:n_samples]
    box_x = f["box_x"][:n_samples]; box_y = f["box_y"][:n_samples]; box_z = f["box_z"][:n_samples]
    n_atoms_after = f["n_atoms_after"][:n_samples]

MATERIAL_TO_IDX = {"Cu": 0, "Fe": 1, "Ni": 2, "Pd": 3, "Pt": 4, "W": 5}
STRUCTURE_TO_IDX = {"bcc": 0, "fcc": 1}
material_idx = np.array([MATERIAL_TO_IDX[m] for m in material_str])
structure_idx = np.array([STRUCTURE_TO_IDX[s] for s in structure_str])

# PKA energy spans 4 orders of magnitude → log1p transform.
# Box volume (~1e5 to ~1e8 Å^3) → log transform. Density is a physical density in atoms/Å^3.
log_pka = np.log1p(pka_energy)
box_vol = box_x * box_y * box_z
log_box = np.log(box_vol)
density = n_atoms_after / box_vol

if EXTRA_FEATURES == "boxdensity":
    continuous = np.stack([lattice_param, log_pka, temperature, log_box, density], axis=1)
elif EXTRA_FEATURES == "none":
    continuous = np.stack([lattice_param, pka_energy, temperature], axis=1)
else:
    raise ValueError(EXTRA_FEATURES)
N_CONTINUOUS = continuous.shape[1]


def make_loader(sel, cont_mean, cont_std, batch_size=128, shuffle=True):
    cont_norm = (continuous[sel] - cont_mean) / cont_std
    return DataLoader(
        TensorDataset(
            torch.tensor(material_idx[sel], dtype=torch.long),
            torch.tensor(structure_idx[sel], dtype=torch.long),
            torch.tensor(cont_norm, dtype=torch.float32),
            torch.tensor(xrd_after[sel], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def make_norm(variant, C):
    if variant == "baseline":
        return nn.BatchNorm1d(C)
    if variant == "no_norm":
        return nn.Identity()
    if variant == "groupnorm":
        g = 1
        for cand in (8, 4, 2, 1):
            if C % cand == 0:
                g = cand
                break
        return nn.GroupNorm(g, C)
    raise ValueError(variant)


class XRDGenerator(nn.Module):
    def __init__(self, variant="baseline", n_continuous=3):
        super().__init__()
        self.mat_emb = nn.Embedding(6, 16)
        self.struct_emb = nn.Embedding(2, 8)
        self.init_c, self.init_l = 32, 32
        input_dim = 16 + 8 + n_continuous
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.init_c * self.init_l),
        )

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

    def forward(self, mat_i, struct_i, cont):
        m = self.mat_emb(mat_i)
        s = self.struct_emb(struct_i)
        x = torch.cat([m, s, cont], dim=1)
        x = self.stem(x).view(-1, self.init_c, self.init_l)
        x = self.decoder(x)
        x = F.interpolate(x, size=500, mode="linear", align_corners=False)
        return x.squeeze(1)


def eval_loss(model, loader, per_mat=False):
    model.eval()
    total = 0.0
    n = 0
    mat_sse = np.zeros(6)
    mat_n = np.zeros(6, dtype=int)
    with torch.no_grad():
        for mat, struct, cont, target in loader:
            mat, struct, cont, target = mat.to(device), struct.to(device), cont.to(device), target.to(device)
            pred = model(mat, struct, cont)
            se = ((pred - target) ** 2).mean(dim=1)  # (batch,)
            total += se.sum().item()
            n += len(mat)
            if per_mat:
                for mi in range(6):
                    sel = (mat == mi)
                    if sel.any():
                        mat_sse[mi] += se[sel].sum().item()
                        mat_n[mi] += int(sel.sum().item())
    mean = total / n
    if per_mat:
        per = np.where(mat_n > 0, mat_sse / np.maximum(mat_n, 1), np.nan)
        return mean, per, mat_n
    return mean


def train_one_fold(train_sel, val_sel, fold_idx):
    cont_mean = continuous[train_sel].mean(axis=0)
    cont_std = continuous[train_sel].std(axis=0) + 1e-8
    train_loader = make_loader(train_sel, cont_mean, cont_std, shuffle=True)
    val_loader = make_loader(val_sel, cont_mean, cont_std, shuffle=False)
    train_eval_loader = make_loader(train_sel, cont_mean, cont_std, shuffle=False)

    model = XRDGenerator(variant=VARIANT, n_continuous=N_CONTINUOUS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    bad = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for mat, struct, cont, target in train_loader:
            mat, struct, cont, target = mat.to(device), struct.to(device), cont.to(device), target.to(device)
            pred = model(mat, struct, cont)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = eval_loss(model, val_loader)
        sched.step(vl)
        if vl < best - 1e-6:
            best = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model = model.to(device)
    tr = eval_loss(model, train_eval_loader)
    vl, per_mat, mat_n = eval_loss(model, val_loader, per_mat=True)
    print(f"  fold {fold_idx}: train={tr:.4f}  val={vl:.4f}  gap={vl-tr:+.4f}  epoch={epoch}")
    return tr, vl, per_mat, mat_n


def main():
    print(f"5-fold CV  variant={VARIANT}  extra={EXTRA_FEATURES}  n_cont={N_CONTINUOUS}  lr={LR}  wd={WD}  epochs={MAX_EPOCHS}  patience={PATIENCE}  seed={SEED}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_evals = []
    val_evals = []
    per_mat_list = []
    mat_n_list = []
    t0 = time.time()
    for fold_idx, (train_sel, val_sel) in enumerate(skf.split(np.arange(n_samples), material_idx)):
        tr, vl, per_mat, mat_n = train_one_fold(train_sel, val_sel, fold_idx)
        train_evals.append(tr)
        val_evals.append(vl)
        per_mat_list.append(per_mat)
        mat_n_list.append(mat_n)
    tr_a = np.array(train_evals)
    vl_a = np.array(val_evals)
    gap = vl_a - tr_a
    print()
    print(f"RESULT variant={VARIANT}  ({time.time()-t0:.0f}s)")
    print(f"  train_eval: mean={tr_a.mean():.4f}  std={tr_a.std():.4f}")
    print(f"  val_eval:   mean={vl_a.mean():.4f}  std={vl_a.std():.4f}")
    print(f"  gap:        mean={gap.mean():+.4f}  std={gap.std():.4f}")
    print(f"  per-material val MSE (mean across folds):")
    per_mat_arr = np.array(per_mat_list)  # (folds, 6)
    for mi, name in [(v, k) for k, v in sorted(MATERIAL_TO_IDX.items(), key=lambda x: x[1])]:
        vals = per_mat_arr[:, mi]
        vals = vals[~np.isnan(vals)]
        print(f"    {name}: {vals.mean():.5f}  (std={vals.std():.5f}, n_total={sum(m[mi] for m in mat_n_list)})")


if __name__ == "__main__":
    main()
