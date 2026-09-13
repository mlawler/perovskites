"""Download a sample of the NOMAD Perovskite Solar Cell Database Project.

Pulls entries via the NOMAD archive/query endpoint with a `required` filter
so only the fields we care about come across the wire, then writes a flat
CSV plus a JSONL of the raw filtered archives.

Target: enough rows to train a composition -> performance surrogate for
transfer learning onto a smaller irradiation dataset later.
"""

import csv
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://nomad-lab.eu/prod/v1/api/v1/entries/archive/query"
DATASET_ID = "bVE6OHGxSxag_i8Y0NC4wQ"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nomad_perovskite"
PAGE_SIZE = 500
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "perovskite_sample"

REQUIRED = {
    "data": {
        "ref": "*",
        "perovskite": "*",
        "perovskite_deposition": "*",
        "jv": "*",
        "stability": "*",
        "cell": "*",
    }
}

CSV_COLS = [
    "entry_id", "doi", "publication_date",
    "composition_long", "composition_short", "band_gap",
    "cell_architecture", "cell_area",
    "deposition_procedure", "deposition_atmosphere",
    "pce", "voc", "jsc", "ff",
    "stability_measured", "stability_atmosphere",
    "stability_relative_humidity", "stability_temperature_range",
    "stability_light_intensity", "stability_PCE_T80",
    "stability_time_total_exposure",
]


def post(body):
    req = Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def flatten(entry):
    a = entry.get("archive", {}).get("data", {})
    ref = a.get("ref", {}) or {}
    pv = a.get("perovskite", {}) or {}
    dep = a.get("perovskite_deposition", {}) or {}
    jv = a.get("jv", {}) or {}
    cell = a.get("cell", {}) or {}
    stab = a.get("stability", {}) or {}
    return {
        "entry_id": entry.get("entry_id"),
        "doi": ref.get("DOI_number") or ref.get("ID"),
        "publication_date": ref.get("publication_date"),
        "composition_long": pv.get("composition_long_form"),
        "composition_short": pv.get("composition_short_form"),
        "band_gap": pv.get("band_gap"),
        "cell_architecture": cell.get("architecture"),
        "cell_area": cell.get("area_measured"),
        "deposition_procedure": dep.get("procedure"),
        "deposition_atmosphere": dep.get("synthesis_atmosphere"),
        "pce": jv.get("default_PCE"),
        "voc": jv.get("default_Voc"),
        "jsc": jv.get("default_Jsc"),
        "ff": jv.get("default_FF"),
        "stability_measured": stab.get("measured"),
        "stability_atmosphere": stab.get("atmosphere"),
        "stability_relative_humidity": stab.get("relative_humidity_range"),
        "stability_temperature_range": stab.get("temperature_range"),
        "stability_light_intensity": stab.get("light_intensity"),
        "stability_PCE_T80": stab.get("PCE_T80"),
        "stability_time_total_exposure": stab.get("time_total_exposure"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUT_NAME}.csv"
    jsonl_path = OUT_DIR / f"{OUT_NAME}.jsonl"
    rows_written = 0
    page_after = None

    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fjson:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        writer.writeheader()
        while rows_written < TARGET:
            pagination = {"page_size": PAGE_SIZE, "order": "asc", "order_by": "entry_id"}
            if page_after:
                pagination["page_after_value"] = page_after
            body = {
                "query": {"datasets.dataset_id": DATASET_ID},
                "pagination": pagination,
                "required": REQUIRED,
            }
            t0 = time.time()
            resp = post(body)
            data = resp.get("data", [])
            pag = resp.get("pagination", {})
            if not data:
                break
            for e in data:
                fjson.write(json.dumps(e) + "\n")
                writer.writerow(flatten(e))
                rows_written += 1
                if rows_written >= TARGET:
                    break
            page_after = pag.get("next_page_after_value")
            total = pag.get("total")
            print(f"  fetched {rows_written}/{TARGET} (of {total} total) in {time.time()-t0:.1f}s", flush=True)
            if not page_after:
                break

    print(f"\nwrote {rows_written} rows -> {csv_path}")
    print(f"raw archives       -> {jsonl_path}")


if __name__ == "__main__":
    sys.exit(main())
