"""Inspect why W samples are ~1000x harder than other metals."""
import os
import h5py
import numpy as np

H5_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cascade_pairs", "cascade_pairs.h5")
with h5py.File(H5_PATH, "r") as f:
    n = int(f.attrs["n_samples"])
    xrd_after = f["xrd_after"][:n]
    xrd_before = f["xrd_before"][:n]
    material_str = [f["material"][i].decode() for i in range(n)]
    lattice_param = f["lattice_param"][:n]
    pka_energy = f["pka_energy"][:n]
    temperature = f["temperature"][:n]
    box_x = f["box_x"][:n]; box_y = f["box_y"][:n]; box_z = f["box_z"][:n]
    n_atoms = f["n_atoms_after"][:n]

material = np.array(material_str)
print(f"Materials and counts:")
for m in np.unique(material):
    mask = material == m
    print(f"  {m}: n={mask.sum()}")

print(f"\nCondition ranges per material:")
print(f"{'mat':>4}  {'n':>5}  {'PKA keV':>22}  {'T K':>22}  {'a Å':>18}  {'box Å':>18}")
for m in np.unique(material):
    mask = material == m
    pk = pka_energy[mask]; tp = temperature[mask]; lp = lattice_param[mask]
    bx = box_x[mask]
    print(f"  {m:>4}  {mask.sum():>5}  "
          f"[{pk.min():>8.1f}, {pk.max():>8.1f}]  "
          f"[{tp.min():>8.1f}, {tp.max():>8.1f}]  "
          f"[{lp.min():>6.3f}, {lp.max():>6.3f}]  "
          f"[{bx.min():>6.1f}, {bx.max():>6.1f}]")

print(f"\nXRD after-spectrum stats per material (S(q) range):")
print(f"{'mat':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'peak_max':>10}")
for m in np.unique(material):
    mask = material == m
    x = xrd_after[mask]
    print(f"  {m:>4}  {x.mean():>8.3f}  {x.std():>8.3f}  {x.min():>8.3f}  {x.max():>8.3f}  {x.max(axis=1).mean():>10.3f}")

print(f"\nPer-material variance of post-damage spectrum (how much does S_after vary within a material?):")
for m in np.unique(material):
    mask = material == m
    x = xrd_after[mask]
    if mask.sum() < 2:
        continue
    # variance across samples of same material, at each q
    v = x.var(axis=0).mean()
    # if the model just predicted the per-material mean, its MSE would equal v
    print(f"  {m}: within-material variance of S(q) = {v:.4f}  (MSE floor for mean-prediction = {v:.4f})")

print(f"\nPKA energy distribution for W (histogram):")
w_pka = pka_energy[material == "W"]
bins = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
hist, _ = np.histogram(w_pka, bins=bins)
for b0, b1, c in zip(bins[:-1], bins[1:], hist):
    print(f"  [{b0:>6}, {b1:>6}) keV: {c:>5}")

print(f"\nTemperature distribution for W:")
w_tp = temperature[material == "W"]
bins = [-1, 0.5, 100, 300, 500, 1000, 2000, 5000]
hist, _ = np.histogram(w_tp, bins=bins)
for b0, b1, c in zip(bins[:-1], bins[1:], hist):
    print(f"  [{b0:>6}, {b1:>6}) K: {c:>5}")
