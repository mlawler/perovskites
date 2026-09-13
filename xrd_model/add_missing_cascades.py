#!/usr/bin/env python3
"""
Add missing W cascade simulations to cascade_pairs.h5.

These sims were skipped in the original build because some atoms had
coordinates outside the simulation box (common in high-energy cascades).
The fix: wrap positions into [0, box) before computing the RDF.

Usage:
    python scripts/add_missing_cascades.py          # dry-run (list missing)
    python scripts/add_missing_cascades.py --run     # process and append
"""

import argparse
import json
import tarfile
import time
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths and constants (same as build_cascade_pairs.py)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "cascadesdb"
H5_PATH = Path(__file__).resolve().parent.parent / "data" / "cascade_pairs" / "cascade_pairs.h5"

RDF_N_BINS = 500
RDF_R_MAX = 10.0
XRD_Q_MIN = 0.5
XRD_Q_MAX = 8.0
XRD_N_Q = 500

BASES = {
    "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    "fcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]),
}

MAX_ATOMS_FULL = 400_000
MAX_CENTERS = 50_000

# ---------------------------------------------------------------------------
# Lattice, RDF, XRD (copied from build_cascade_pairs.py, with wrapping fix)
# ---------------------------------------------------------------------------

def generate_perfect_lattice(structure, a, box_lengths):
    basis = BASES[structure]
    nx = int(round(box_lengths[0] / a))
    ny = int(round(box_lengths[1] / a))
    nz = int(round(box_lengths[2] / a))
    grid = np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T
    positions = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    return positions


def wrap_positions(positions, box):
    """Wrap positions into [0, box) — the fix for the original build errors."""
    box = np.asarray(box, dtype=np.float64)
    pos = positions % box
    # Clamp to avoid floating-point edge cases where pos == box exactly
    eps = box * 1e-12
    return np.clip(pos, 0.0, box - eps)


def compute_rdf(positions, box, n_bins=RDF_N_BINS, r_max=RDF_R_MAX):
    n = len(positions)
    box = np.asarray(box, dtype=np.float64)
    density = n / np.prod(box)
    pos = wrap_positions(positions, box)

    bin_edges = np.linspace(0, r_max, n_bins + 1)

    if n <= MAX_ATOMS_FULL:
        tree = cKDTree(pos, boxsize=box)
        pairs = tree.query_pairs(bin_edges[-1], output_type="ndarray")
        hist = np.zeros(len(bin_edges) - 1, dtype=np.float64)
        CHUNK = 5_000_000
        for start in range(0, len(pairs), CHUNK):
            chunk = pairs[start:start + CHUNK]
            dr = pos[chunk[:, 0]] - pos[chunk[:, 1]]
            dr -= box * np.round(dr / box)
            dists = np.linalg.norm(dr, axis=1)
            h, _ = np.histogram(dists, bins=bin_edges)
            hist += h
        g_r = (2.0 * hist) / (n * density * _shell_volumes(bin_edges))
    else:
        M = min(n, MAX_CENTERS)
        rng = np.random.default_rng(42)
        center_idx = rng.choice(n, M, replace=False)
        center_pos = pos[center_idx]
        tree_all = cKDTree(pos, boxsize=box)
        hist = np.zeros(len(bin_edges) - 1, dtype=np.float64)
        CHUNK = 2000
        for start in range(0, M, CHUNK):
            end = min(start + CHUNK, M)
            chunk_pos = center_pos[start:end]
            chunk_global = center_idx[start:end]
            nbr_lists = tree_all.query_ball_point(chunk_pos, bin_edges[-1])
            all_centers = []
            all_nbrs = []
            for local_i, nbrs in enumerate(nbr_lists):
                nbrs = np.asarray(nbrs, dtype=np.int64)
                nbrs = nbrs[nbrs != chunk_global[local_i]]
                if len(nbrs) == 0:
                    continue
                all_centers.append(np.full(len(nbrs), local_i, dtype=np.int64))
                all_nbrs.append(nbrs)
            if not all_nbrs:
                continue
            center_local = np.concatenate(all_centers)
            nbr_global = np.concatenate(all_nbrs)
            dr = pos[nbr_global] - chunk_pos[center_local]
            dr -= box * np.round(dr / box)
            dists = np.linalg.norm(dr, axis=1)
            h, _ = np.histogram(dists, bins=bin_edges)
            hist += h
        g_r = hist / (M * density * _shell_volumes(bin_edges))

    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return r, g_r


def _shell_volumes(bin_edges):
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]
    return 4.0 * np.pi * r**2 * dr


def compute_xrd(r, g_r, density, q_min=XRD_Q_MIN, q_max=XRD_Q_MAX, n_q=XRD_N_Q):
    q = np.linspace(q_min, q_max, n_q)
    dr = r[1] - r[0]
    r_max = r[-1] + 0.5 * dr
    lorch = np.where(r > 0, np.sin(np.pi * r / r_max) / (np.pi * r / r_max), 1.0)
    base = r**2 * (g_r - 1.0) * lorch * dr
    qr = q[:, None] * r[None, :]
    sinc_qr = np.where(qr > 0, np.sin(qr) / qr, 1.0)
    S_q = 1.0 + 4.0 * np.pi * density * (sinc_qr @ base)

    err_lorch = _lorch_error_bound(r, g_r, density, lorch, q, dr)
    err_trunc = _truncation_error_bound(r, g_r, density, dr)

    return q, S_q, err_lorch, err_trunc


def _lorch_error_bound(r, g_r, density, lorch, q, dr):
    """Per-q upper bound on |S_Lorch(q) - S_trunc(q)|."""
    weight = np.abs(r**2 * (g_r - 1.0) * (lorch - 1.0)) * dr
    scalar_bound = 4.0 * np.pi * density * np.sum(weight)

    qr = q[:, None] * r[None, :]
    sinc_qr = np.where(qr > 0, np.sin(qr) / qr, 1.0)
    q_bound = 4.0 * np.pi * density * np.abs(sinc_qr @ (r**2 * (g_r - 1.0) * (lorch - 1.0) * dr))

    return {"scalar": float(scalar_bound), "per_q": q_bound}


def _truncation_error_bound(r, g_r, density, dr):
    """Estimate truncation error from the tail of g(r)-1 beyond r_max."""
    n = len(r)
    tail_start = int(0.9 * n)
    tail_r = r[tail_start:]
    tail_gr = g_r[tail_start:]
    deviation = tail_gr - 1.0

    tail_rms = float(np.sqrt(np.mean(deviation**2)))
    tail_integral = float(
        4.0 * np.pi * density * np.sum(tail_r**2 * np.abs(deviation) * dr)
    )

    return {"tail_rms": tail_rms, "tail_integral_last_10pct": tail_integral}


# ---------------------------------------------------------------------------
# XYZ parsing
# ---------------------------------------------------------------------------

def read_xyz_from_tar(tar, member_name):
    f = tar.extractfile(member_name)
    raw = f.read()
    lines = raw.split(b"\n")
    n_atoms = int(lines[0].strip())
    comment = lines[1].decode().strip()
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        parts = lines[i + 2].split()
        positions[i, 0] = float(parts[1])
        positions[i, 1] = float(parts[2])
        positions[i, 2] = float(parts[3])
    return n_atoms, comment, positions


def parse_box_from_comment(comment):
    parts = comment.split()
    for i, p in enumerate(parts):
        if p.lower().startswith("boxsize"):
            return [float(parts[i + 1]), float(parts[i + 2]), float(parts[i + 3])]
    return None


# ---------------------------------------------------------------------------
# Before-state cache
# ---------------------------------------------------------------------------

class BeforeCache:
    def __init__(self):
        self._cache = {}

    def get(self, structure, a, box):
        key = (structure, round(a, 6), tuple(round(b, 4) for b in box))
        if key in self._cache:
            return self._cache[key]
        pos = generate_perfect_lattice(structure, a, box)
        n = len(pos)
        density = n / (box[0] * box[1] * box[2])
        r, g_r = compute_rdf(pos, box)
        q, s_q, err_lorch, err_trunc = compute_xrd(r, g_r, density)
        result = (n, r, g_r.astype(np.float32), q, s_q.astype(np.float32),
                  err_lorch, err_trunc)
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Find missing simulations
# ---------------------------------------------------------------------------

KNOWN_MISSING_ARCHIVES = [
    "025-T1025K_50keV_data.tar.gz",
    "036-T2050K_75keV_data.tar.gz",
    "0fc-363K_35keV_100Direction.tar.bz2",
    "102-363K_35keV_110Direction.tar.bz2",
    "108-363K_35keV_111Direction.tar.bz2",
    "10f-363K_35keV_122Direction.tar.bz2",
    "115-363K_35keV_133Direction.tar.bz2",
    "11c-363K_35keV_135Direction.tar.bz2",
    "123-363K_35keV_235Direction.tar.bz2",
    # Partially missing (1-3 sims each):
    "101-363K_20keV_110Direction.tar.bz2",
    "103-363K_50keV_110Direction.tar.bz2",
    "109-363K_50keV_111Direction.tar.bz2",
    "10a-363K_80keV_111Direction.tar.bz2",
    "10d-363K_10keV_122Direction.tar.bz2",
    "110-363K_50keV_122Direction.tar.bz2",
    "116-363K_50keV_133Direction.tar.bz2",
    "117-363K_80keV_133Direction.tar.bz2",
    "11a-363K_10keV_135Direction.tar.bz2",
    "11e-363K_80keV_135Direction.tar.bz2",
    "124-363K_50keV_235Direction.tar.bz2",
    "125-363K_80keV_235Direction.tar.bz2",
]


def find_missing():
    """Return list of (tar_path, xyz_name, metadata) for unprocessed perfect-crystal W sims."""
    # Load already-processed pairs
    with h5py.File(H5_PATH, "r") as f:
        processed = set()
        n = int(f.attrs["n_samples"])
        for i in range(n):
            processed.add((f["archive_name"][i].decode(), f["source_file"][i].decode()))

    # Build metadata lookup
    meta_dir = BASE_DIR / "metadata"
    stem_to_meta = {}
    for mf in sorted(meta_dir.glob("*.json")):
        with open(mf) as f:
            m = json.load(f)
        aname = m.get("archive-name", "")
        stem = aname.replace(".tar.gz", "").replace(".tar.bz2", "")
        stem_to_meta[stem] = m

    # Only scan the specific archives known to have missing sims
    archive_dir = BASE_DIR / "archives" / "W"
    missing = []
    for aname in KNOWN_MISSING_ARCHIVES:
        tar_path = archive_dir / aname
        if not tar_path.exists():
            print(f"  Warning: {aname} not found", file=sys.stderr)
            continue
        stem = aname.replace(".tar.gz", "").replace(".tar.bz2", "")
        if stem not in stem_to_meta:
            print(f"  Warning: no metadata for {aname}", file=sys.stderr)
            continue
        meta = stem_to_meta[stem]
        try:
            with tarfile.open(tar_path) as t:
                xyz_members = [
                    m.name for m in t.getmembers()
                    if m.name.endswith(".xyz")
                    and "/._" not in m.name
                    and not m.name.startswith("._")
                ]
                for xyz in xyz_members:
                    if (tar_path.name, xyz) not in processed:
                        missing.append((tar_path, xyz, meta))
        except Exception as e:
            print(f"  Cannot open {aname}: {e}", file=sys.stderr)

    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Actually process (default is dry-run)")
    args = parser.parse_args()

    print("Scanning for missing perfect-crystal simulations...")
    missing = find_missing()
    print(f"Found {len(missing)} missing simulations\n")

    if not missing:
        print("Nothing to do.")
        return

    # Group by archive for display
    by_archive = {}
    for tar_path, xyz, meta in missing:
        by_archive.setdefault(tar_path.name, []).append(xyz)
    for aname, xyzs in sorted(by_archive.items()):
        print(f"  {aname}: {len(xyzs)} sims")

    if not args.run:
        print(f"\nDry run. Use --run to process and append to {H5_PATH}")
        return

    # Open HDF5 for appending
    f = h5py.File(H5_PATH, "a")
    n = int(f.attrs["n_samples"])
    print(f"\nAppending to {H5_PATH} (currently {n} samples)")

    before_cache = BeforeCache()
    added = 0
    errors = 0
    t_start = time.time()

    current_tar = None
    current_tar_path = None

    for idx, (tar_path, xyz_name, meta) in enumerate(missing):
        mat = meta["material"]["chemical-formula"]
        struct = meta["material"]["structure"]
        a = meta["material"]["lattice-parameters"]["a"]
        pka_energy = meta["PKA-energy"]
        temperature = meta.get("initial-temperature", 0.0)
        box_meta = [
            meta["simulation-box"]["box-X-length"],
            meta["simulation-box"]["box-Y-length"],
            meta["simulation-box"]["box-Z-length"],
        ]

        # Reuse open tar if same archive
        if tar_path != current_tar_path:
            if current_tar is not None:
                current_tar.close()
            current_tar = tarfile.open(tar_path)
            current_tar_path = tar_path

        try:
            t0 = time.time()
            n_atoms, comment, pos_after = read_xyz_from_tar(current_tar, xyz_name)
            box_after = parse_box_from_comment(comment)
            if box_after is None:
                box_after = list(box_meta)

            # Before state (cached perfect lattice)
            (n_before, r, g_r_before, q, s_q_before,
             err_lorch_before, err_trunc_before) = before_cache.get(struct, a, box_after)
            if n_before != n_atoms:
                (n_before, r, g_r_before, q, s_q_before,
                 err_lorch_before, err_trunc_before) = before_cache.get(struct, a, box_meta)

            # After state (with position wrapping fix)
            box = np.array(box_after)
            density_after = n_atoms / np.prod(box)
            r, g_r_after = compute_rdf(pos_after, box)
            q, s_q_after, err_lorch_after, err_trunc_after = compute_xrd(
                r, g_r_after, density_after
            )
            del pos_after

            # Append to HDF5
            i = n + added
            for key in f:
                if key in ("r", "q"):
                    continue
                ds = f[key]
                new_shape = list(ds.shape)
                new_shape[0] = i + 1
                ds.resize(new_shape)

            f["rdf_before"][i] = g_r_before
            f["rdf_after"][i] = g_r_after.astype(np.float32)
            f["xrd_before"][i] = s_q_before
            f["xrd_after"][i] = s_q_after.astype(np.float32)
            f["material"][i] = mat
            f["structure"][i] = struct
            f["archive_name"][i] = tar_path.name
            f["source_file"][i] = xyz_name
            f["lattice_param"][i] = a
            f["pka_energy"][i] = pka_energy
            f["temperature"][i] = temperature
            f["n_atoms_before"][i] = n_before
            f["n_atoms_after"][i] = n_atoms
            f["box_x"][i] = box[0]
            f["box_y"][i] = box[1]
            f["box_z"][i] = box[2]
            f["err_lorch_scalar_before"][i] = err_lorch_before["scalar"]
            f["err_lorch_scalar_after"][i] = err_lorch_after["scalar"]
            f["err_lorch_per_q_before"][i] = err_lorch_before["per_q"].astype(np.float32)
            f["err_lorch_per_q_after"][i] = err_lorch_after["per_q"].astype(np.float32)
            f["err_trunc_tail_rms_before"][i] = err_trunc_before["tail_rms"]
            f["err_trunc_tail_rms_after"][i] = err_trunc_after["tail_rms"]
            f["err_trunc_tail_integral_before"][i] = err_trunc_before["tail_integral_last_10pct"]
            f["err_trunc_tail_integral_after"][i] = err_trunc_after["tail_integral_last_10pct"]

            added += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            rate = added / total_elapsed if total_elapsed > 0 else 0
            remaining = (len(missing) - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx+1}/{len(missing)}] {tar_path.name}/{xyz_name}: "
                f"{n_atoms:,} atoms, {elapsed:.1f}s "
                f"({added} added, ~{remaining:.0f}s remaining)"
            )

        except Exception as e:
            errors += 1
            print(f"  [{idx+1}/{len(missing)}] {xyz_name}: ERROR: {e}", file=sys.stderr)
            import gc
            gc.collect()

        # Flush periodically
        if added % 10 == 0:
            f.attrs["n_samples"] = n + added
            f.flush()

    if current_tar is not None:
        current_tar.close()

    f.attrs["n_samples"] = n + added
    f.close()

    total_time = time.time() - t_start
    print(f"\nDone. Added {added} samples ({errors} errors) in {total_time:.0f}s")
    print(f"Dataset now has {n + added} total samples")


if __name__ == "__main__":
    main()
