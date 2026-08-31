"""Download + cache TCGA-BRCA clinical metadata and gene expression.

Sources:

- GDC TCGA-BRCA ``/cases`` endpoint for survival and diagnosis fields used by
  the ER+ survival panel.
- cBioPortal Datahub ``brca_tcga_pub2015`` sample clinical table for ER status
  used by TCGA stratification.
- cBioPortal Datahub ``brca_tcga_pub2015`` RSEM gene-expression matrix for
  regulator-expression associations.

Cached structure:

- ``<data_root>/tcga_brca_ankrd30a/tcga_brca_gdc_clinical.tsv`` stores a flat
  GDC clinical table with OS fields.
- ``<data_root>/tcga_brca_ankrd30a/tcga_brca_gdc_cases_clinical.raw.json`` and
  ``gdc_status.json`` retain the source GDC responses for provenance.
- ``<data_root>/tcga_brca_ankrd30a/brca_tcga_pub2015_data_clinical_sample.txt``
  stores the raw cBioPortal sample clinical table.
- ``<data_root>/tcga_brca_ankrd30a/brca_tcga_pub2015_data_mrna_seq_v2_rsem.txt``
  stores the raw cBioPortal gene-level RSEM expression matrix.

Idempotent pipeline:

1. Reuses every non-empty local file unless ``--force`` is supplied.
2. Downloads cBioPortal clinical/expression files with plain HTTP GET.
3. Queries GDC ``/status`` and ``/cases`` and writes one compact flat TSV.
4. Does not write filtered regulator-expression or receptor-status intermediates;
   ANKRD30A analysis derives those in memory from the raw data files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.atera_ffpe_human_brca.ankrd30a"

from scripts.paper_utils import log

from ..config import (
    TCGA_CBIO_CLINICAL_SAMPLE_TXT,
    TCGA_CBIO_RSEM_TXT,
    TCGA_DATA_DIR,
    TCGA_GDC_CLINICAL_RAW_JSON,
    TCGA_GDC_CLINICAL_TSV,
    TCGA_GDC_STATUS_JSON,
)
from ..utils import project_path

DEFAULT_GDC_API_BASE = "https://api.gdc.cancer.gov"
CBIO_CLINICAL_SAMPLE_URL = (
    "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/"
    "brca_tcga_pub2015/data_clinical_sample.txt"
)
CBIO_RSEM_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
    "brca_tcga_pub2015/data_mrna_seq_v2_rsem.txt"
)

GDC_CLINICAL_FIELDS = [
    "case_id",
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.year_of_birth",
    "demographic.year_of_death",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_grade",
    "diagnoses.classification_of_tumor",
    "diagnoses.prior_malignancy",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy",
    "follow_ups.days_to_follow_up",
    "follow_ups.progression_or_recurrence",
    "follow_ups.disease_response",
    "follow_ups.days_to_recurrence",
]

GDC_CLINICAL_COLUMNS = [
    "case_id",
    "case_submitter_id",
    "vital_status",
    "os_event",
    "os_days",
    "days_to_death",
    "days_to_last_follow_up",
    "gender",
    "race",
    "ethnicity",
    "year_of_birth",
    "year_of_death",
    "age_at_diagnosis_days",
    "age_at_diagnosis_years",
    "ajcc_pathologic_stage",
    "ajcc_pathologic_t",
    "ajcc_pathologic_n",
    "ajcc_pathologic_m",
    "primary_diagnosis",
    "tumor_grade",
    "classification_of_tumor",
    "prior_malignancy",
    "morphology",
    "tissue_or_organ_of_origin",
    "site_of_resection_or_biopsy",
    "n_diagnoses",
    "n_follow_ups",
    "max_days_to_recurrence",
    "follow_up_disease_response",
    "progression_or_recurrence",
]


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _write_tsv(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    log(f"wrote {project_path(path)} ({len(rows):,} rows)")


def download_url(url: str, path: Path, *, force: bool = False) -> Path:
    """Download one public URL unless a non-empty local copy already exists."""
    if _nonempty(path) and not force:
        log(f"reuse {project_path(path)}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sonic-paper/atera"})
    with urllib.request.urlopen(request, timeout=120) as response:
        # Use an atomic rename so partial network downloads are never reused.
        with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    log(f"downloaded {url} -> {project_path(path)}")
    return path


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    """POST or GET JSON from GDC."""
    if payload is None:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GDC request failed: HTTP {exc.code}: {detail}") from exc


def _to_float(value: object) -> float:
    try:
        if value in {None, ""}:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _max_numeric(values: Iterable[object]) -> float:
    vals = [_to_float(value) for value in values]
    vals = [value for value in vals if math.isfinite(value)]
    return max(vals) if vals else math.nan


def _first_nonempty(values: Iterable[object]) -> object:
    for value in values:
        if value not in {None, ""}:
            return value
    return ""


def flatten_gdc_case(hit: dict[str, object]) -> dict[str, object]:
    """Flatten one GDC case record into the clinical fields needed by figures."""
    diagnoses = hit.get("diagnoses") if isinstance(hit.get("diagnoses"), list) else []
    follow_ups = hit.get("follow_ups") if isinstance(hit.get("follow_ups"), list) else []
    demographic = hit.get("demographic") if isinstance(hit.get("demographic"), dict) else {}
    dx0 = diagnoses[0] if diagnoses and isinstance(diagnoses[0], dict) else {}

    vital_status = demographic.get("vital_status")
    days_to_death = _to_float(demographic.get("days_to_death"))
    dx_last_follow = _max_numeric(
        dx.get("days_to_last_follow_up") for dx in diagnoses if isinstance(dx, dict)
    )
    fu_last_follow = _max_numeric(
        fu.get("days_to_follow_up") for fu in follow_ups if isinstance(fu, dict)
    )
    # OS time is death time for events and the latest follow-up time otherwise.
    last_follow = _max_numeric([dx_last_follow, fu_last_follow])
    event = int(str(vital_status).lower() == "dead")
    os_days = days_to_death if event and math.isfinite(days_to_death) else last_follow
    age_days = _to_float(dx0.get("age_at_diagnosis"))

    return {
        "case_id": hit.get("case_id", ""),
        "case_submitter_id": hit.get("submitter_id", ""),
        "vital_status": vital_status or "",
        "os_event": event,
        "os_days": os_days,
        "days_to_death": days_to_death,
        "days_to_last_follow_up": last_follow,
        "gender": demographic.get("gender", ""),
        "race": demographic.get("race", ""),
        "ethnicity": demographic.get("ethnicity", ""),
        "year_of_birth": demographic.get("year_of_birth", ""),
        "year_of_death": demographic.get("year_of_death", ""),
        "age_at_diagnosis_days": age_days,
        "age_at_diagnosis_years": (age_days / 365.25 if math.isfinite(age_days) else math.nan),
        "ajcc_pathologic_stage": dx0.get("ajcc_pathologic_stage", ""),
        "ajcc_pathologic_t": dx0.get("ajcc_pathologic_t", ""),
        "ajcc_pathologic_n": dx0.get("ajcc_pathologic_n", ""),
        "ajcc_pathologic_m": dx0.get("ajcc_pathologic_m", ""),
        "primary_diagnosis": dx0.get("primary_diagnosis", ""),
        "tumor_grade": dx0.get("tumor_grade", ""),
        "classification_of_tumor": dx0.get("classification_of_tumor", ""),
        "prior_malignancy": dx0.get("prior_malignancy", ""),
        "morphology": dx0.get("morphology", ""),
        "tissue_or_organ_of_origin": dx0.get("tissue_or_organ_of_origin", ""),
        "site_of_resection_or_biopsy": dx0.get("site_of_resection_or_biopsy", ""),
        "n_diagnoses": len(diagnoses),
        "n_follow_ups": len(follow_ups),
        "max_days_to_recurrence": _max_numeric(
            fu.get("days_to_recurrence") for fu in follow_ups if isinstance(fu, dict)
        ),
        "follow_up_disease_response": ";".join(
            sorted(
                {
                    str(fu.get("disease_response"))
                    for fu in follow_ups
                    if isinstance(fu, dict) and fu.get("disease_response")
                }
            )
        ),
        "progression_or_recurrence": _first_nonempty(
            fu.get("progression_or_recurrence") for fu in follow_ups if isinstance(fu, dict)
        ),
    }


def fetch_gdc_clinical(*, force: bool = False, api_base: str = DEFAULT_GDC_API_BASE) -> Path:
    """Fetch GDC TCGA-BRCA case-level clinical metadata into the Atera data cache."""
    if _nonempty(TCGA_GDC_CLINICAL_TSV) and _nonempty(TCGA_GDC_CLINICAL_RAW_JSON) and not force:
        log(f"reuse {project_path(TCGA_GDC_CLINICAL_TSV)}")
        return TCGA_GDC_CLINICAL_TSV

    TCGA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The status payload is not used analytically; it records the GDC API state
    # that produced the clinical table.
    status = request_json(f"{api_base.rstrip('/')}/status")
    TCGA_GDC_STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "project.project_id", "value": ["TCGA-BRCA"]},
        },
        "fields": ",".join(GDC_CLINICAL_FIELDS),
        "format": "JSON",
        "size": 2000,
    }
    response = request_json(f"{api_base.rstrip('/')}/cases", payload)
    TCGA_GDC_CLINICAL_RAW_JSON.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
    hits = response.get("data", {}).get("hits", [])
    rows = [flatten_gdc_case(hit) for hit in hits if isinstance(hit, dict)]
    _write_tsv(rows, TCGA_GDC_CLINICAL_TSV, GDC_CLINICAL_COLUMNS)
    return TCGA_GDC_CLINICAL_TSV


def ensure_tcga_clinical_expression_data(
    *,
    force: bool = False,
    include_expression: bool = True,
    include_gdc_clinical: bool = True,
    gdc_api_base: str = DEFAULT_GDC_API_BASE,
) -> dict[str, str]:
    """Ensure TCGA clinical/expression source files exist, returning path statuses."""
    TCGA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    status["cbio_clinical_sample"] = str(
        download_url(CBIO_CLINICAL_SAMPLE_URL, TCGA_CBIO_CLINICAL_SAMPLE_TXT, force=force)
    )
    if include_expression:
        status["cbio_rsem"] = str(download_url(CBIO_RSEM_URL, TCGA_CBIO_RSEM_TXT, force=force))
    if include_gdc_clinical:
        status["gdc_clinical"] = str(fetch_gdc_clinical(force=force, api_base=gdc_api_base))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TCGA-BRCA clinical and gene-expression inputs."
    )
    parser.add_argument("--force", action="store_true", help="Redownload existing non-empty files.")
    parser.add_argument(
        "--skip-expression",
        action="store_true",
        help="Skip cBioPortal RSEM expression matrix.",
    )
    parser.add_argument(
        "--skip-gdc-clinical",
        action="store_true",
        help="Skip GDC case-level clinical metadata.",
    )
    parser.add_argument("--gdc-api-base", default=DEFAULT_GDC_API_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_tcga_clinical_expression_data(
        force=args.force,
        include_expression=not args.skip_expression,
        include_gdc_clinical=not args.skip_gdc_clinical,
        gdc_api_base=args.gdc_api_base,
    )


if __name__ == "__main__":
    main()
