#!/usr/bin/env python3
"""
Build before/after XRD and RDF dataset from CascadesDB cascade simulations.

Each simulation starts from a perfect crystal (before) and produces a
radiation-damaged configuration (after). This script computes RDF g(r)
and powder structure factor S(q) for both states and stores everything
in HDF5 for PyTorch consumption.

Usage:
    python scripts/build_cascade_pairs.py            # full dataset
    python scripts/build_cascade_pairs.py --samples   # sample archives only
    python scripts/build_cascade_pairs.py --resume    # resume interrupted run
"""

import argparse
import json
import tarfile
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "cascadesdb"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cascade_pairs"

RDF_N_BINS = 500
RDF_R_MAX = 10.0
XRD_Q_MIN = 0.5
XRD_Q_MAX = 8.0
XRD_N_Q = 500


# ---------------------------------------------------------------------------
# Lattice generation
# ---------------------------------------------------------------------------

BASES = {
    "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    "fcc": np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    ),
}


def generate_perfect_lattice(structure, a, box_lengths):
    basis = BASES[structure]
    nx = int(round(box_lengths[0] / a))
    ny = int(round(box_lengths[1] / a))
    nz = int(round(box_lengths[2] / a))
    grid = np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T
    positions = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    return positions


# ---------------------------------------------------------------------------
# RDF computation
# ---------------------------------------------------------------------------

MAX_ATOMS_FULL = 400_000
MAX_CENTERS = 50_000


def compute_rdf(positions, box, n_bins=RDF_N_BINS, r_max=RDF_R_MAX):
    n = len(positions)
    box = np.asarray(box, dtype=np.float64)
    density = n / np.prod(box)
    pos = positions % box

    bin_edges = np.linspace(0, r_max, n_bins + 1)

    if n <= MAX_ATOMS_FULL:
        hist = _rdf_full(pos, box, bin_edges)
        g_r = (2.0 * hist) / (n * density * _shell_volumes(bin_edges))
    else:
        M = min(n, MAX_CENTERS)
        hist = _rdf_sampled(pos, box, bin_edges, M)
        g_r = hist / (M * density * _shell_volumes(bin_edges))

    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return r, g_r


def _shell_volumes(bin_edges):
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]
    return 4.0 * np.pi * r**2 * dr


def _rdf_full(pos, box, bin_edges):
    """Exact RDF via query_pairs — fast for systems up to ~400K atoms."""
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(bin_edges[-1], output_type="ndarray")

    CHUNK = 5_000_000
    hist = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    for start in range(0, len(pairs), CHUNK):
        chunk = pairs[start : start + CHUNK]
        dr = pos[chunk[:, 0]] - pos[chunk[:, 1]]
        dr -= box * np.round(dr / box)
        dists = np.linalg.norm(dr, axis=1)
        h, _ = np.histogram(dists, bins=bin_edges)
        hist += h
    return hist


def _rdf_sampled(pos, box, bin_edges, M):
    """Memory-efficient RDF: pick M center atoms, find all their neighbors."""
    n = len(pos)
    rng = np.random.default_rng(42)
    center_idx = rng.choice(n, M, replace=False)
    center_pos = pos[center_idx]
    r_max = bin_edges[-1]

    tree_all = cKDTree(pos, boxsize=box)
    hist = np.zeros(len(bin_edges) - 1, dtype=np.float64)

    CHUNK = 2000
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        chunk_pos = center_pos[start:end]
        chunk_global = center_idx[start:end]

        nbr_lists = tree_all.query_ball_point(chunk_pos, r_max)

        # Flatten all neighbor lists into one vectorized distance computation
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

    return hist


# ---------------------------------------------------------------------------
# XRD from RDF via Debye relation
# ---------------------------------------------------------------------------

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
    """Per-q upper bound on |S_Lorch(q) - S_trunc(q)|.

    The Lorch window modifies the integrand by the factor W(r). The error
    relative to the unwindowed (but still truncated) integral is:

        eps(q) = 4*pi*rho * |sum_r r^2 [g(r)-1] [W(r)-1] sinc(qr) dr|

    We bound |sinc(qr)| <= 1 to get a q-independent scalar bound, and also
    compute the full q-dependent bound.
    """
    weight = np.abs(r**2 * (g_r - 1.0) * (lorch - 1.0)) * dr
    scalar_bound = 4.0 * np.pi * density * np.sum(weight)

    qr = q[:, None] * r[None, :]
    sinc_qr = np.where(qr > 0, np.sin(qr) / qr, 1.0)
    q_bound = 4.0 * np.pi * density * np.abs(sinc_qr @ (r**2 * (g_r - 1.0) * (lorch - 1.0) * dr))

    return {"scalar": float(scalar_bound), "per_q": q_bound}


def _truncation_error_bound(r, g_r, density, dr):
    """Estimate truncation error from the tail of g(r)-1 beyond r_max.

    We cannot compute the missing integral directly (we don't have g(r)
    beyond r_max), but we can estimate it from the behavior of g(r) near
    r_max. If g(r)-1 has not decayed to zero by r_max, the residual
    amplitude tells us roughly how large the missing contribution is.

    We report:
      tail_amplitude: RMS of g(r)-1 in the last 10% of the r range
      tail_integral:  4*pi*rho * integral of r^2*|g(r)-1| dr over last 10%
                      (per unit r, this estimates the density of missing signal)
    """
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
# Fast XYZ parsing
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
# Archive discovery
# ---------------------------------------------------------------------------

def find_all_archives():
    """Find all 320 archives in archives/ matched to metadata."""
    meta_dir = BASE_DIR / "metadata"
    archive_dir = BASE_DIR / "archives"

    stem_to_meta = {}
    for mf in sorted(meta_dir.glob("*.json")):
        with open(mf) as f:
            m = json.load(f)
        aname = m.get("archive-name", "")
        stem = aname.replace(".tar.gz", "").replace(".tar.bz2", "")
        stem_to_meta[stem] = (mf.name, m)

    archives = []
    for mat_dir in sorted(archive_dir.iterdir()):
        if not mat_dir.is_dir():
            continue
        for tar_path in sorted(mat_dir.glob("*.tar.*")):
            stem = tar_path.name.replace(".tar.gz", "").replace(".tar.bz2", "")
            if stem in stem_to_meta:
                meta_name, metadata = stem_to_meta[stem]
                archives.append((tar_path, meta_name, metadata))

    return archives


def find_sample_archives():
    """Find the top-level sample archives."""
    archives = []
    for tar_path in sorted(BASE_DIR.glob("*.tar.bz2")):
        meta_path = tar_path.with_name(
            tar_path.name.replace(".tar.bz2", "_metadata.json")
        )
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            archives.append((tar_path, meta_path.name, metadata))
    return archives


# ---------------------------------------------------------------------------
# Before-state cache
# ---------------------------------------------------------------------------

class BeforeCache:
    """Cache RDF/XRD for perfect lattices — keyed by (structure, a, box)."""

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
# HDF5 streaming writer
# ---------------------------------------------------------------------------

class H5Writer:
    def __init__(self, path, resume=False):
        self.path = path
        if resume and path.exists():
            self.f = h5py.File(path, "a")
            self.n = int(self.f.attrs.get("n_samples", 0))
            self.processed = set()
            if "source_file" in self.f and "archive_name" in self.f:
                for i in range(self.n):
                    key = (
                        self.f["archive_name"][i].decode(),
                        self.f["source_file"][i].decode(),
                    )
                    self.processed.add(key)
            print(f"Resuming: {self.n} samples already written, "
                  f"{len(self.processed)} tracked")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = h5py.File(path, "w")
            self.n = 0
            self.processed = set()
            self._init_datasets()

    def _init_datasets(self):
        f = self.f
        r_centers = np.linspace(0, RDF_R_MAX, RDF_N_BINS + 1)
        r_centers = 0.5 * (r_centers[:-1] + r_centers[1:])
        q_centers = np.linspace(XRD_Q_MIN, XRD_Q_MAX, XRD_N_Q)

        f.create_dataset("r", data=r_centers.astype(np.float32))
        f["r"].attrs["units"] = "Angstrom"
        f.create_dataset("q", data=q_centers.astype(np.float32))
        f["q"].attrs["units"] = "1/Angstrom"

        maxshape = (None,)
        maxshape2 = (None, RDF_N_BINS)
        maxshape2q = (None, XRD_N_Q)
        chunks2 = (min(100, 17024), RDF_N_BINS)
        chunks2q = (min(100, 17024), XRD_N_Q)
        chunks1 = (min(1000, 17024),)

        dt_str = h5py.string_dtype()
        f.create_dataset("rdf_before", shape=(0, RDF_N_BINS), maxshape=maxshape2,
                         dtype=np.float32, chunks=chunks2)
        f.create_dataset("rdf_after", shape=(0, RDF_N_BINS), maxshape=maxshape2,
                         dtype=np.float32, chunks=chunks2)
        f.create_dataset("xrd_before", shape=(0, XRD_N_Q), maxshape=maxshape2q,
                         dtype=np.float32, chunks=chunks2q)
        f.create_dataset("xrd_after", shape=(0, XRD_N_Q), maxshape=maxshape2q,
                         dtype=np.float32, chunks=chunks2q)

        for name, dtype in [
            ("material", dt_str),
            ("structure", dt_str),
            ("archive_name", dt_str),
            ("source_file", dt_str),
        ]:
            f.create_dataset(name, shape=(0,), maxshape=maxshape, dtype=dtype,
                             chunks=chunks1)

        for name in [
            "lattice_param", "pka_energy", "temperature",
            "box_x", "box_y", "box_z",
        ]:
            f.create_dataset(name, shape=(0,), maxshape=maxshape,
                             dtype=np.float32, chunks=chunks1)

        for name in ["n_atoms_before", "n_atoms_after"]:
            f.create_dataset(name, shape=(0,), maxshape=maxshape,
                             dtype=np.int32, chunks=chunks1)

        for name in [
            "err_lorch_scalar_before", "err_lorch_scalar_after",
            "err_trunc_tail_rms_before", "err_trunc_tail_rms_after",
            "err_trunc_tail_integral_before", "err_trunc_tail_integral_after",
        ]:
            f.create_dataset(name, shape=(0,), maxshape=maxshape,
                             dtype=np.float32, chunks=chunks1)

        for name in [
            "err_lorch_per_q_before", "err_lorch_per_q_after",
        ]:
            f.create_dataset(name, shape=(0, XRD_N_Q), maxshape=maxshape2q,
                             dtype=np.float32, chunks=chunks2q)

        f.attrs["description"] = (
            "Before/after RDF and XRD pairs from CascadesDB collision cascade "
            "simulations. 'Before' is the perfect crystal reconstructed from "
            "metadata. 'After' is the post-cascade configuration."
        )
        f.attrs["rdf_r_max"] = RDF_R_MAX
        f.attrs["rdf_n_bins"] = RDF_N_BINS
        f.attrs["xrd_q_min"] = XRD_Q_MIN
        f.attrs["xrd_q_max"] = XRD_Q_MAX
        f.attrs["xrd_n_q"] = XRD_N_Q
        f.attrs["n_samples"] = 0

    def is_processed(self, archive_name, source_file):
        return (archive_name, source_file) in self.processed

    def append(self, record):
        i = self.n
        for key in self.f:
            if key in ("r", "q"):
                continue
            ds = self.f[key]
            new_shape = list(ds.shape)
            new_shape[0] = i + 1
            ds.resize(new_shape)

        self.f["rdf_before"][i] = record["rdf_before"]
        self.f["rdf_after"][i] = record["rdf_after"]
        self.f["xrd_before"][i] = record["xrd_before"]
        self.f["xrd_after"][i] = record["xrd_after"]
        self.f["material"][i] = record["material"]
        self.f["structure"][i] = record["structure"]
        self.f["archive_name"][i] = record["archive_name"]
        self.f["source_file"][i] = record["source_file"]
        self.f["lattice_param"][i] = record["lattice_param"]
        self.f["pka_energy"][i] = record["pka_energy"]
        self.f["temperature"][i] = record["temperature"]
        self.f["n_atoms_before"][i] = record["n_atoms_before"]
        self.f["n_atoms_after"][i] = record["n_atoms_after"]
        self.f["box_x"][i] = record["box_x"]
        self.f["box_y"][i] = record["box_y"]
        self.f["box_z"][i] = record["box_z"]
        self.f["err_lorch_scalar_before"][i] = record["err_lorch_scalar_before"]
        self.f["err_lorch_scalar_after"][i] = record["err_lorch_scalar_after"]
        self.f["err_lorch_per_q_before"][i] = record["err_lorch_per_q_before"]
        self.f["err_lorch_per_q_after"][i] = record["err_lorch_per_q_after"]
        self.f["err_trunc_tail_rms_before"][i] = record["err_trunc_tail_rms_before"]
        self.f["err_trunc_tail_rms_after"][i] = record["err_trunc_tail_rms_after"]
        self.f["err_trunc_tail_integral_before"][i] = record["err_trunc_tail_integral_before"]
        self.f["err_trunc_tail_integral_after"][i] = record["err_trunc_tail_integral_after"]

        self.n = i + 1
        self.f.attrs["n_samples"] = self.n
        self.processed.add((record["archive_name"], record["source_file"]))

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.attrs["n_samples"] = self.n
        self.f.close()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_archive(tar_path, archive_name, metadata, writer, before_cache):
    mat = metadata["material"]["chemical-formula"]
    struct = metadata["material"]["structure"]
    a = metadata["material"]["lattice-parameters"]["a"]
    pka_energy = metadata["PKA-energy"]
    temperature = metadata.get("initial-temperature", 0.0)
    meta_box = metadata["simulation-box"]
    box_meta = [
        meta_box["box-X-length"],
        meta_box["box-Y-length"],
        meta_box["box-Z-length"],
    ]

    tar = tarfile.open(tar_path)
    xyz_members = sorted(
        [m.name for m in tar.getmembers()
         if m.name.endswith(".xyz") and "/._" not in m.name
         and not m.name.startswith("._")]
    )

    processed = 0
    skipped = 0
    errors = 0
    for i, xyz_name in enumerate(xyz_members):
        if writer.is_processed(archive_name, xyz_name):
            skipped += 1
            continue

        try:
            t0 = time.time()
            n_atoms, comment, pos_after = read_xyz_from_tar(tar, xyz_name)
            box_after = parse_box_from_comment(comment)
            if box_after is None:
                box_after = list(box_meta)
            t_parse = time.time() - t0

            # Before state (cached)
            t0 = time.time()
            (n_before, r, g_r_before, q, s_q_before,
             err_lorch_before, err_trunc_before) = before_cache.get(
                struct, a, box_after
            )
            if n_before != n_atoms:
                (n_before, r, g_r_before, q, s_q_before,
                 err_lorch_before, err_trunc_before) = before_cache.get(
                    struct, a, box_meta
                )
            t_before = time.time() - t0

            # After state
            t0 = time.time()
            box = np.array(box_after)
            density_after = n_atoms / np.prod(box)
            r, g_r_after = compute_rdf(pos_after, box)
            q, s_q_after, err_lorch_after, err_trunc_after = compute_xrd(
                r, g_r_after, density_after
            )
            t_after = time.time() - t0

            del pos_after

            writer.append({
                "material": mat,
                "structure": struct,
                "lattice_param": a,
                "pka_energy": pka_energy,
                "temperature": temperature,
                "n_atoms_before": n_before,
                "n_atoms_after": n_atoms,
                "box_x": box[0],
                "box_y": box[1],
                "box_z": box[2],
                "rdf_before": g_r_before,
                "rdf_after": g_r_after.astype(np.float32),
                "xrd_before": s_q_before,
                "xrd_after": s_q_after.astype(np.float32),
                "err_lorch_scalar_before": err_lorch_before["scalar"],
                "err_lorch_scalar_after": err_lorch_after["scalar"],
                "err_lorch_per_q_before": err_lorch_before["per_q"].astype(np.float32),
                "err_lorch_per_q_after": err_lorch_after["per_q"].astype(np.float32),
                "err_trunc_tail_rms_before": err_trunc_before["tail_rms"],
                "err_trunc_tail_rms_after": err_trunc_after["tail_rms"],
                "err_trunc_tail_integral_before": err_trunc_before["tail_integral_last_10pct"],
                "err_trunc_tail_integral_after": err_trunc_after["tail_integral_last_10pct"],
                "archive_name": archive_name,
                "source_file": xyz_name,
            })
            processed += 1

            total_t = t_parse + t_before + t_after
            print(
                f"    [{i+1}/{len(xyz_members)}] {xyz_name}: "
                f"{n_atoms:,} atoms, {total_t:.1f}s "
                f"(parse={t_parse:.1f} before={t_before:.1f} after={t_after:.1f})"
            )
        except Exception as e:
            errors += 1
            print(f"    [{i+1}/{len(xyz_members)}] {xyz_name}: ERROR: {e}",
                  file=sys.stderr)
            import gc
            gc.collect()

    tar.close()

    if skipped:
        print(f"    (skipped {skipped} already processed)")
    if errors:
        print(f"    ({errors} errors)")

    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true",
                        help="Only process sample archives")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    args = parser.parse_args()

    if args.samples:
        archives = find_sample_archives()
        print(f"Sample mode: {len(archives)} archives")
    else:
        archives = find_all_archives()
        print(f"Full mode: {len(archives)} archives")

    if not archives:
        print("No archives found.", file=sys.stderr)
        sys.exit(1)

    total_sims = sum(
        m.get("number-of-simulations", 1) for _, _, m in archives
    )
    print(f"Total expected simulations: {total_sims}")

    h5_path = OUT_DIR / "cascade_pairs.h5"
    writer = H5Writer(h5_path, resume=args.resume)
    before_cache = BeforeCache()

    t_start = time.time()
    total_processed = 0

    for arch_idx, (tar_path, meta_name, metadata) in enumerate(archives):
        mat = metadata["material"]["chemical-formula"]
        energy = metadata["PKA-energy"]
        temp = metadata.get("initial-temperature", 0.0)
        n_sims = metadata.get("number-of-simulations", "?")

        elapsed = time.time() - t_start
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = (total_sims - writer.n) / rate / 3600 if rate > 0 else 0

        print(
            f"\n[{arch_idx+1}/{len(archives)}] {mat} {energy} keV {temp} K "
            f"({n_sims} sims) — {tar_path.name}"
        )
        print(
            f"  Progress: {writer.n}/{total_sims} done, "
            f"{rate:.1f} sims/s, ~{remaining:.1f}h remaining"
        )

        try:
            n = process_archive(
                tar_path, tar_path.name, metadata, writer, before_cache
            )
            total_processed += n
        except Exception as e:
            print(f"  ARCHIVE ERROR: {e}", file=sys.stderr)

        writer.flush()

    writer.close()

    elapsed = time.time() - t_start
    size_mb = h5_path.stat().st_size / 1e6
    print(f"\nDone. {writer.n} pairs in {h5_path} ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")

    # Summary
    with h5py.File(h5_path, "r") as f:
        materials = [m.decode() for m in f["material"][:]]
    from collections import Counter
    print("\nBy material:")
    for mat, count in sorted(Counter(materials).items()):
        print(f"  {mat}: {count}")


if __name__ == "__main__":
    main()
