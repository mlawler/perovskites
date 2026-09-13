"""
Build before/during/after HDF5 dataset from NOMAD perovskite stability data.

Each sample has:
  BEFORE:  PCE, Voc, Jsc, FF  (4 floats — fresh device performance)
  DURING:  exposure_time, atmosphere, temperature, light_intensity,
           light_source, bias_condition, uv_filter  (7 features — stress conditions)
  AFTER:   pce_retention  (1 float — % of initial PCE at end of experiment)

Missing values: NaN for numeric, "" for categorical.

Also stores composition and metadata for each sample.
"""

import json
import math
import sys

import h5py
import numpy as np


JSONL_PATH = "data/nomad_perovskite/perovskite_full.jsonl"
OUT_PATH = "data/perovskite_pairs/perovskite_pairs.h5"

A_SITE_IONS = ["Cs", "FA", "MA", "BA", "PEA", "Rb", "K", "GA", "DMA", "EDA"]
B_SITE_IONS = ["Pb", "Sn", "Ge", "Bi", "Sb", "Cu", "Mn", "Ti"]
C_SITE_IONS = ["I", "Br", "Cl", "SCN", "BF4", "F"]

ATMOSPHERE_VOCAB = ["Air", "N2", "Ambient", "Dry air", "Ar", "Vacuum"]
LIGHT_SOURCE_VOCAB = ["Dark", "Light", "Xenon", "Solar simulator", "White LED", "UV lamp",
                       "Sulfur plasma", "Halogen", "Metal halide"]
BIAS_VOCAB = ["Open circuit", "MPPT", "Constant potential", "Short circuit", "Passive resistance"]


def parse_ion_coefficients(ions_str, coeff_str, vocab):
    """Parse ';'-separated ion names and coefficients into a fixed-length vector."""
    vec = np.full(len(vocab), 0.0, dtype=np.float32)
    if not ions_str or not coeff_str:
        return vec
    ions_str = str(ions_str).strip()
    coeff_str = str(coeff_str).strip()
    if ions_str in ("nan", "Unknown", ""):
        return vec

    # Handle multi-layer compositions like "MA | (MIC3)" — take first layer
    if "|" in ions_str:
        ions_str = ions_str.split("|")[0].strip()
        coeff_str = coeff_str.split("|")[0].strip()

    ions = [s.strip() for s in ions_str.split(";")]
    coeffs = [s.strip() for s in coeff_str.split(";")]

    for ion, coeff in zip(ions, coeffs):
        if ion in vocab:
            try:
                vec[vocab.index(ion)] = float(coeff)
            except (ValueError, TypeError):
                pass
    return vec


def parse_temperature(temp_str):
    """Parse 'lo; hi' temperature range, return midpoint."""
    if not temp_str or str(temp_str) in ("nan", "nan; nan", "Unknown", ""):
        return float("nan")
    parts = str(temp_str).split(";")
    try:
        vals = [float(p.strip()) for p in parts if p.strip() not in ("nan", "")]
        return np.mean(vals) if vals else float("nan")
    except ValueError:
        return float("nan")


def encode_categorical(value, vocab):
    """Encode categorical as integer index (0-based), -1 for missing/unknown."""
    s = str(value).strip() if value is not None else ""
    if s in ("nan", "Unknown", "None", ""):
        return -1
    if s in vocab:
        return vocab.index(s)
    return -1


def main():
    records = []
    n_total = 0
    n_no_before = 0
    n_no_after = 0

    with open(JSONL_PATH) as f:
        for line in f:
            n_total += 1
            rec = json.loads(line)
            data = rec["archive"]["data"]
            stab = data.get("stability", {})
            jv = data.get("jv", {})
            perov = data.get("perovskite", {})
            ref = data.get("ref", {})
            cell = data.get("cell", {})

            # -- BEFORE: need at least PCE --
            pce = jv.get("default_PCE")
            voc = jv.get("default_Voc")
            jsc = jv.get("default_Jsc")
            ff = jv.get("default_FF")
            if not all(v is not None for v in [pce, voc, jsc, ff]):
                n_no_before += 1
                continue

            # -- AFTER: need PCE retention --
            pce_end = stab.get("PCE_end_of_experiment")
            if pce_end is None or str(pce_end) == "nan":
                n_no_after += 1
                continue
            try:
                pce_retention = float(pce_end)
            except (ValueError, TypeError):
                n_no_after += 1
                continue

            # -- DURING: extract all, NaN/empty for missing --
            time_val = stab.get("time_total_exposure")
            try:
                exposure_time = float(time_val) if time_val is not None and str(time_val) not in ("nan", "Unknown") else float("nan")
            except (ValueError, TypeError):
                exposure_time = float("nan")

            temperature = parse_temperature(stab.get("temperature_range"))

            light_val = stab.get("light_intensity")
            try:
                light_intensity = float(light_val) if light_val is not None and str(light_val) not in ("nan", "Unknown", "None") else float("nan")
            except (ValueError, TypeError):
                light_intensity = float("nan")

            atmosphere_idx = encode_categorical(stab.get("atmosphere"), ATMOSPHERE_VOCAB)
            light_source_idx = encode_categorical(stab.get("light_source_type"), LIGHT_SOURCE_VOCAB)
            bias_idx = encode_categorical(stab.get("potential_bias_load_condition"), BIAS_VOCAB)

            uv_filter = stab.get("light_UV_filter")
            if uv_filter is None:
                uv_val = float("nan")
            else:
                uv_val = 1.0 if uv_filter else 0.0

            # -- Composition --
            a_vec = parse_ion_coefficients(
                perov.get("composition_a_ions"),
                perov.get("composition_a_ions_coefficients"),
                A_SITE_IONS,
            )
            b_vec = parse_ion_coefficients(
                perov.get("composition_b_ions"),
                perov.get("composition_b_ions_coefficients"),
                B_SITE_IONS,
            )
            c_vec = parse_ion_coefficients(
                perov.get("composition_c_ions"),
                perov.get("composition_c_ions_coefficients"),
                C_SITE_IONS,
            )

            band_gap = perov.get("band_gap")
            try:
                band_gap = float(band_gap) if band_gap is not None and str(band_gap) not in ("nan", "Unknown", "") else float("nan")
            except (ValueError, TypeError):
                band_gap = float("nan")

            comp_long = str(perov.get("composition_long_form", ""))
            if "|" in comp_long:
                comp_long = comp_long.split("|")[0].strip()

            records.append({
                "entry_id": rec.get("entry_id", ""),
                "doi": str(ref.get("DOI_number", "")),
                "composition": comp_long,
                "architecture": str(cell.get("architecture", "")),
                # before
                "pce": float(pce),
                "voc": float(voc),
                "jsc": float(jsc),
                "ff": float(ff),
                "band_gap": band_gap,
                # composition vectors
                "a_site": a_vec,
                "b_site": b_vec,
                "c_site": c_vec,
                # during
                "exposure_time": exposure_time,
                "atmosphere_idx": atmosphere_idx,
                "temperature": temperature,
                "light_intensity": light_intensity,
                "light_source_idx": light_source_idx,
                "bias_idx": bias_idx,
                "uv_filter": uv_val,
                # after
                "pce_retention": pce_retention,
            })

    n = len(records)
    print(f"Total JSONL records: {n_total}")
    print(f"Skipped (no before): {n_no_before}")
    print(f"Skipped (no after):  {n_no_after}")
    print(f"Valid samples:       {n}")

    # -- Write HDF5 --
    with h5py.File(OUT_PATH, "w") as f:
        f.attrs["n_samples"] = n
        f.attrs["a_site_ions"] = A_SITE_IONS
        f.attrs["b_site_ions"] = B_SITE_IONS
        f.attrs["c_site_ions"] = C_SITE_IONS
        f.attrs["atmosphere_vocab"] = ATMOSPHERE_VOCAB
        f.attrs["light_source_vocab"] = LIGHT_SOURCE_VOCAB
        f.attrs["bias_vocab"] = BIAS_VOCAB

        dt_str = h5py.string_dtype()

        # String datasets
        for key in ["entry_id", "doi", "composition", "architecture"]:
            f.create_dataset(key, data=[r[key] for r in records], dtype=dt_str)

        # Before scalars
        for key in ["pce", "voc", "jsc", "ff", "band_gap"]:
            f.create_dataset(key, data=np.array([r[key] for r in records], dtype=np.float32))

        # Composition vectors
        f.create_dataset("a_site_coeffs", data=np.stack([r["a_site"] for r in records]))
        f.create_dataset("b_site_coeffs", data=np.stack([r["b_site"] for r in records]))
        f.create_dataset("c_site_coeffs", data=np.stack([r["c_site"] for r in records]))

        # During — continuous
        for key in ["exposure_time", "temperature", "light_intensity", "uv_filter"]:
            f.create_dataset(key, data=np.array([r[key] for r in records], dtype=np.float32))

        # During — categorical (integer-coded, -1 = missing)
        for key in ["atmosphere_idx", "light_source_idx", "bias_idx"]:
            f.create_dataset(key, data=np.array([r[key] for r in records], dtype=np.int32))

        # After
        f.create_dataset("pce_retention", data=np.array([r["pce_retention"] for r in records], dtype=np.float32))

    print(f"\nWrote {n} samples to {OUT_PATH}")

    # Summary stats
    pce_ret = np.array([r["pce_retention"] for r in records])
    exp_t = np.array([r["exposure_time"] for r in records])
    valid_t = exp_t[~np.isnan(exp_t)]
    n_complete = sum(
        1 for r in records
        if not math.isnan(r["exposure_time"])
        and r["atmosphere_idx"] >= 0
        and not math.isnan(r["temperature"])
        and not math.isnan(r["light_intensity"])
        and r["light_source_idx"] >= 0
        and r["bias_idx"] >= 0
    )
    print(f"\nPCE retention: mean={np.mean(pce_ret):.1f}%, median={np.median(pce_ret):.1f}%, std={np.std(pce_ret):.1f}%")
    print(f"Exposure time: mean={np.mean(valid_t):.0f}h, median={np.median(valid_t):.0f}h (of {len(valid_t)} with time)")
    print(f"Complete during features: {n_complete}/{n}")


if __name__ == "__main__":
    main()
