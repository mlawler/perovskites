"""Resume downloading the NOMAD Perovskite Solar Cell Database from a given entry_id."""

import csv
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://nomad-lab.eu/prod/v1/api/v1/entries/archive/query"
DATASET_ID = "bVE6OHGxSxag_i8Y0NC4wQ"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nomad_perovskite"
PAGE_SIZE = 200
RESUME_AFTER = sys.argv[1] if len(sys.argv) > 1 else None
TARGET_TOTAL = 43119

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
    csv_path = OUT_DIR / "perovskite_full.csv"
    jsonl_path = OUT_DIR / "perovskite_full.jsonl"
    page_after = RESUME_AFTER
    rows_appended = 0
    retries = 0
    max_retries = 5

    with open(csv_path, "a", newline="") as fcsv, open(jsonl_path, "a") as fjson:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLS)
        while True:
            pagination = {"page_size": PAGE_SIZE, "order": "asc", "order_by": "entry_id"}
            if page_after:
                pagination["page_after_value"] = page_after
            body = {
                "query": {"datasets.dataset_id": DATASET_ID},
                "pagination": pagination,
                "required": REQUIRED,
            }
            t0 = time.time()
            try:
                resp = post(body)
                retries = 0
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"  FATAL: {e} after {max_retries} retries", flush=True)
                    break
                wait = min(30 * retries, 120)
                print(f"  retry {retries}/{max_retries} after error: {e} (waiting {wait}s)", flush=True)
                time.sleep(wait)
                continue
            data = resp.get("data", [])
            pag = resp.get("pagination", {})
            if not data:
                break
            for e in data:
                fjson.write(json.dumps(e) + "\n")
                writer.writerow(flatten(e))
                rows_appended += 1
            page_after = pag.get("next_page_after_value")
            total = pag.get("total")
            elapsed = time.time() - t0
            print(f"  +{len(data)} (appended {rows_appended}, total ~{24000+rows_appended}/{total}) in {elapsed:.1f}s", flush=True)
            fcsv.flush()
            fjson.flush()
            if not page_after:
                break

    print(f"\nAppended {rows_appended} rows to {csv_path}")


if __name__ == "__main__":
    main()
