"""Download all CascadesDB archives from the IAEA cascade simulation database."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "data" / "cascadesdb" / "catalog.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cascadesdb" / "archives"
BASE_URL = "https://cascadesdb.iaea.org"
MAX_RETRIES = 3


def download(url, dest, timeout=600):
    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), "-o", str(dest),
             "-w", "%{http_code}", url],
            capture_output=True, text=True,
        )
        code = result.stdout.strip()
        if code == "200" and dest.exists() and dest.stat().st_size > 0:
            return True
        if attempt < MAX_RETRIES:
            wait = 10 * attempt
            print(f"    retry {attempt}/{MAX_RETRIES} (HTTP {code}), waiting {wait}s", flush=True)
            time.sleep(wait)
    return False


def main():
    records = json.load(open(CATALOG))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also fetch metadata JSONs
    meta_dir = OUT_DIR.parent / "metadata"
    meta_dir.mkdir(exist_ok=True)

    total = len(records)
    done = 0
    skipped = 0
    failed = []
    total_bytes = 0
    t_start = time.time()

    for i, r in enumerate(records):
        archive_path = r.get("archive")
        if not archive_path:
            skipped += 1
            continue

        fname = archive_path.split("/")[-1]
        mat = r["material"].split()[0]
        energy = r["energy_keV"]
        rid = r["id"]

        # Organize by material subdirectory
        mat_dir = OUT_DIR / mat
        mat_dir.mkdir(exist_ok=True)
        dest = mat_dir / fname

        if dest.exists() and dest.stat().st_size > 1000:
            done += 1
            total_bytes += dest.stat().st_size
            continue

        url = BASE_URL + archive_path
        t0 = time.time()
        ok = download(url, dest)
        elapsed = time.time() - t0

        if ok:
            sz = dest.stat().st_size
            total_bytes += sz
            done += 1
            rate = sz / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(
                f"  [{done}/{total}] {mat} {energy}keV id={rid}: "
                f"{fname} ({sz/1024/1024:.1f} MB, {rate:.1f} MB/s)",
                flush=True,
            )
        else:
            failed.append(r)
            print(f"  [{i+1}/{total}] FAILED: {fname}", flush=True)

        # Fetch metadata JSON
        meta_file = meta_dir / f"{rid}_{mat}_{energy}keV.json"
        if not meta_file.exists():
            subprocess.run(
                ["curl", "-sS", "--max-time", "20", "-o", str(meta_file),
                 f"{BASE_URL}/cdbmeta/cdbrecord/json/{rid}/"],
                capture_output=True,
            )

    wall = time.time() - t_start
    print(f"\nDone in {wall/60:.1f} min")
    print(f"Downloaded: {done}/{total} archives ({total_bytes/1024/1024/1024:.1f} GB)")
    print(f"Skipped (no URL): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  {r['material']} {r['energy_keV']}keV id={r['id']}")


if __name__ == "__main__":
    main()
