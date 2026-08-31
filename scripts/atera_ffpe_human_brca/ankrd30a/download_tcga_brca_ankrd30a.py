"""Download + cache TCGA-BRCA RNA-seq BAM slices around ANKRD30A.

Source: Genomic Data Commons (GDC) TCGA-BRCA controlled-access RNA-seq BAMs.
The default query targets aligned RNA-seq BAM files from the STAR 2-pass genome
workflow and slices the hg38 ANKRD30A locus ``chr10:37115598-37242567``
for the PacBio-only transcript and direct Atera probe-region validation workflow.

Cached structure:

- ``<data_root>/tcga_brca_ankrd30a/tcga_brca_rnaseq_bam_manifest_with_samples.tsv``
  stores GDC file metadata, case/sample identifiers, duplicate flags, slice
  paths, and per-file slice status.
- ``<data_root>/tcga_brca_ankrd30a/tcga_brca_rnaseq_bam_manifest_validation.json``
  stores a compact manifest/slicing summary.
- ``<data_root>/tcga_brca_ankrd30a/bam_slices/*.bam`` stores per-file ANKRD30A
  locus slices, with matching ``.bai`` and ``.idxstats.tsv`` files.
- ``<data_root>/tcga_brca_ankrd30a/logs/`` stores ``samtools`` validation and
  indexing logs.

Idempotent pipeline:

1. Queries the GDC ``/files`` endpoint and writes the local manifest. If the
   manifest already exists, it is reused unless ``--force`` is supplied.
2. Downloads each controlled BAM slice through the GDC slicing endpoint using
   ``GDC_TOKEN``, ``GDC_TOKEN_FILE``, or ``--token-file``.
3. Runs ``samtools quickcheck``, ``samtools index``, and ``samtools idxstats``
   for each slice and records the result back into the manifest.
4. Reuses existing non-empty BAM slices unless ``--force`` is supplied, so
   interrupted runs can resume from the current manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.atera_ffpe_human_brca.ankrd30a"

from scripts.paper_utils import log

from ..config import (
    TCGA_BAM_MANIFEST_TSV,
    TCGA_BAM_SLICE_DIR,
    TCGA_DATA_DIR,
)
from .common import DEFAULT_SLICE_REGION

DEFAULT_GDC_API_BASE = "https://api.gdc.cancer.gov"
DEFAULT_PROJECT = "TCGA-BRCA"
DEFAULT_GENE = "ANKRD30A"
DEFAULT_GENCODE_RELEASE = "49"
DEFAULT_WORKFLOW_TYPE = "STAR 2-Pass Genome"

GDC_FIELDS = [
    "file_id",
    "file_name",
    "data_type",
    "data_format",
    "experimental_strategy",
    "access",
    "cases.case_id",
    "cases.submitter_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.portions.analytes.aliquots.submitter_id",
    "associated_entities.entity_submitter_id",
    "associated_entities.entity_type",
    "analysis.workflow_type",
]

MANIFEST_COLUMNS = [
    "file_id",
    "bam_file_name",
    "data_type",
    "data_format",
    "experimental_strategy",
    "access",
    "workflow_type",
    "case_submitter_id",
    "case_id",
    "sample_submitter_id",
    "sample_id",
    "sample_type",
    "aliquot_submitter_id",
    "associated_entity_submitter_id",
    "associated_entity_type",
    "n_case_submitter_ids",
    "n_sample_submitter_ids",
    "n_aliquot_submitter_ids",
    "n_files_for_case_submitter_id",
    "n_files_for_sample_submitter_id",
    "duplicate_case_submitter_id",
    "duplicate_sample_submitter_id",
    "slice_region",
    "local_slice_bam",
    "slice_status",
    "slice_bytes",
    "slice_error",
]


def sanitize_token(value: object, *, fallback: str = "NA") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback))
    return text.strip("_") or fallback


def semijoin(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return ";".join(out)


def split_semis(value: str | None) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def read_token(token_file: Path | None) -> str | None:
    # CLI token wins, then environment file, then raw environment token.
    if token_file:
        return token_file.expanduser().read_text().strip()
    if os.environ.get("GDC_TOKEN_FILE"):
        return Path(os.environ["GDC_TOKEN_FILE"]).expanduser().read_text().strip()
    if os.environ.get("GDC_TOKEN"):
        return os.environ["GDC_TOKEN"].strip()
    return None


def write_tsv(
    rows: list[dict[str, object]], path: Path, fieldnames: list[str] = MANIFEST_COLUMNS
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    log(f"wrote {path} ({len(rows):,} rows)")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log(f"wrote {path}")


def in_filter(field: str, values: list[str]) -> dict[str, object]:
    return {"op": "in", "content": {"field": field, "value": values}}


def gdc_request_json(
    api_base: str, payload: dict[str, object], token: str | None
) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["X-Auth-Token"] = token
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/files",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GDC request failed: HTTP {exc.code}: {detail}") from exc


def build_filters(args: argparse.Namespace) -> dict[str, object]:
    # Restrict the manifest to TCGA-BRCA RNA-seq BAMs unless the CLI broadens it.
    content = [
        in_filter("cases.project.project_id", [args.project]),
        in_filter("data_type", ["Aligned Reads"]),
        in_filter("data_format", ["BAM"]),
        in_filter("experimental_strategy", ["RNA-Seq"]),
    ]
    if args.controlled_only:
        content.append(in_filter("access", ["controlled"]))
    if not args.all_workflows:
        content.append(
            in_filter("analysis.workflow_type", args.workflow_type or [DEFAULT_WORKFLOW_TYPE])
        )
    return {"op": "and", "content": content}


def flatten_hit(hit: dict[str, object], *, slice_region: str) -> dict[str, str]:
    # GDC nests case/sample/aliquot identifiers; flatten them into stable
    # semicolon-delimited manifest columns for downstream joins.
    case_ids: list[str] = []
    case_submitters: list[str] = []
    sample_ids: list[str] = []
    sample_submitters: list[str] = []
    sample_types: list[str] = []
    aliquot_submitters: list[str] = []

    for case in hit.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_ids.append(str(case.get("case_id", "")))
        case_submitters.append(str(case.get("submitter_id", "")))
        for sample in case.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            sample_ids.append(str(sample.get("sample_id", "")))
            sample_submitters.append(str(sample.get("submitter_id", "")))
            sample_types.append(str(sample.get("sample_type", "")))
            for portion in sample.get("portions") or []:
                for analyte in portion.get("analytes") or [] if isinstance(portion, dict) else []:
                    for aliquot in (
                        analyte.get("aliquots") or [] if isinstance(analyte, dict) else []
                    ):
                        if isinstance(aliquot, dict):
                            aliquot_submitters.append(str(aliquot.get("submitter_id", "")))

    associated = hit.get("associated_entities") or []
    if not isinstance(associated, list):
        associated = []
    analysis = hit.get("analysis") if isinstance(hit.get("analysis"), dict) else {}

    case_submitter_id = semijoin(case_submitters)
    sample_submitter_id = semijoin(sample_submitters)
    aliquot_submitter_id = semijoin(aliquot_submitters)
    file_id = str(hit.get("file_id", ""))
    local_name = "_".join(
        [
            sanitize_token(case_submitter_id),
            sanitize_token(sample_submitter_id),
            sanitize_token(aliquot_submitter_id),
            sanitize_token(file_id),
            f"{DEFAULT_GENE}.gencode_v{DEFAULT_GENCODE_RELEASE}",
        ]
    )

    return {
        "file_id": file_id,
        "bam_file_name": str(hit.get("file_name", "")),
        "data_type": str(hit.get("data_type", "")),
        "data_format": str(hit.get("data_format", "")),
        "experimental_strategy": str(hit.get("experimental_strategy", "")),
        "access": str(hit.get("access", "")),
        "workflow_type": str(analysis.get("workflow_type", "")),
        "case_submitter_id": case_submitter_id,
        "case_id": semijoin(case_ids),
        "sample_submitter_id": sample_submitter_id,
        "sample_id": semijoin(sample_ids),
        "sample_type": semijoin(sample_types),
        "aliquot_submitter_id": aliquot_submitter_id,
        "associated_entity_submitter_id": semijoin(
            x.get("entity_submitter_id") for x in associated if isinstance(x, dict)
        ),
        "associated_entity_type": semijoin(
            x.get("entity_type") for x in associated if isinstance(x, dict)
        ),
        "n_case_submitter_ids": str(len(split_semis(case_submitter_id))),
        "n_sample_submitter_ids": str(len(split_semis(sample_submitter_id))),
        "n_aliquot_submitter_ids": str(len(split_semis(aliquot_submitter_id))),
        "slice_region": slice_region,
        "local_slice_bam": str(TCGA_BAM_SLICE_DIR / f"{local_name}.bam"),
        "slice_status": "not_started",
        "slice_bytes": "",
        "slice_error": "",
    }


def add_duplicate_flags(rows: list[dict[str, str]]) -> None:
    # Keep duplicates visible because TCGA can have multiple BAM files per case/sample.
    case_counts = Counter(
        row.get("case_submitter_id", "") for row in rows if row.get("case_submitter_id")
    )
    sample_counts = Counter(
        row.get("sample_submitter_id", "") for row in rows if row.get("sample_submitter_id")
    )
    for row in rows:
        case_key = row.get("case_submitter_id", "")
        sample_key = row.get("sample_submitter_id", "")
        row["n_files_for_case_submitter_id"] = str(case_counts.get(case_key, 0))
        row["n_files_for_sample_submitter_id"] = str(sample_counts.get(sample_key, 0))
        row["duplicate_case_submitter_id"] = "true" if case_counts.get(case_key, 0) > 1 else "false"
        row["duplicate_sample_submitter_id"] = (
            "true" if sample_counts.get(sample_key, 0) > 1 else "false"
        )


def validation_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "n_files": len(rows),
        "n_unique_cases": len(
            {row.get("case_submitter_id", "") for row in rows if row.get("case_submitter_id")}
        ),
        "n_unique_samples": len(
            {row.get("sample_submitter_id", "") for row in rows if row.get("sample_submitter_id")}
        ),
        "n_ok_slices": sum(row.get("slice_status") == "ok" for row in rows),
        "n_error_slices": sum(row.get("slice_status") == "error" for row in rows),
    }


def query_manifest(args: argparse.Namespace) -> Path:
    TCGA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TCGA_BAM_SLICE_DIR.mkdir(parents=True, exist_ok=True)
    if TCGA_BAM_MANIFEST_TSV.exists() and not args.force:
        log(f"reuse manifest {TCGA_BAM_MANIFEST_TSV}")
        return TCGA_BAM_MANIFEST_TSV
    token = read_token(args.token_file)
    rows: list[dict[str, str]] = []
    total: int | None = None
    offset = 0

    # Page through /files so large manifests do not depend on a single response.
    while total is None or offset < total:
        payload = {
            "filters": build_filters(args),
            "fields": ",".join(GDC_FIELDS),
            "format": "JSON",
            "size": args.page_size,
            "from": offset,
            "sort": "file_name:asc",
        }
        data = gdc_request_json(args.gdc_api_base, payload, token)
        block = data.get("data") if isinstance(data, dict) else {}
        if not isinstance(block, dict):
            raise RuntimeError(f"Unexpected GDC response: {data}")
        hits = block.get("hits") or []
        total = int((block.get("pagination") or {}).get("total", len(hits)))
        log(f"GDC files page from={offset:,}: {len(hits):,} hits of {total:,}")
        for hit in hits:
            if isinstance(hit, dict):
                rows.append(flatten_hit(hit, slice_region=args.slice_region))
            if args.max_files and len(rows) >= args.max_files:
                total = len(rows)
                break
        offset += args.page_size
        if args.max_files and len(rows) >= args.max_files:
            rows = rows[: args.max_files]
            break

    add_duplicate_flags(rows)
    write_tsv(rows, TCGA_BAM_MANIFEST_TSV)
    write_json(
        validation_summary(rows),
        TCGA_DATA_DIR / "tcga_brca_rnaseq_bam_manifest_validation.json",
    )
    return TCGA_BAM_MANIFEST_TSV


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"run: {' '.join(cmd)}")
    with log_path.open("w") as handle:
        completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit {completed.returncode}: {' '.join(cmd)}")


def download_bam_slice(args: argparse.Namespace, file_id: str, output_bam: Path) -> None:
    if output_bam.exists() and output_bam.stat().st_size > 0 and not args.force:
        log(f"reuse {output_bam}")
        return
    token = read_token(args.token_file)
    if not token:
        raise SystemExit(
            "A GDC token is required for controlled BAM slicing. Pass --token-file or set GDC_TOKEN_FILE/GDC_TOKEN."
        )
    output_bam.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"region": args.slice_region})
    # GDC slicing returns only the ANKRD30A locus, keeping controlled data local footprint small.
    request = urllib.request.Request(
        f"{args.gdc_api_base.rstrip('/')}/slicing/view/{file_id}?{query}",
        headers={"X-Auth-Token": token},
    )
    with urllib.request.urlopen(request) as response:
        with tempfile.NamedTemporaryFile("wb", dir=str(output_bam.parent), delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(output_bam)
    log(f"downloaded slice {file_id} {args.slice_region} -> {output_bam}")


def slice_bams(args: argparse.Namespace) -> Path:
    rows = read_tsv(TCGA_BAM_MANIFEST_TSV)
    rows_to_process = rows[: args.max_files] if args.max_files else rows
    samtools = args.samtools or shutil.which("samtools")
    if not samtools:
        raise SystemExit("Required executable not found on PATH: samtools")
    logs = TCGA_DATA_DIR / "logs"

    for idx, row in enumerate(rows_to_process, start=1):
        output_bam = Path(row["local_slice_bam"])
        row["slice_region"] = args.slice_region
        log(f"slice {idx:,}/{len(rows_to_process):,}: {row['file_id']}")
        download_bam_slice(args, row["file_id"], output_bam)
        # Validate/index every slice before marking it usable by quantification.
        run_cmd(
            [samtools, "quickcheck", "-v", str(output_bam)],
            logs / f"{output_bam.name}.quickcheck.log",
        )
        run_cmd(
            [samtools, "index", "-@", str(args.threads), str(output_bam)],
            logs / f"{output_bam.name}.index.log",
        )
        with output_bam.with_suffix(output_bam.suffix + ".idxstats.tsv").open("w") as handle:
            completed = subprocess.run(
                [samtools, "idxstats", str(output_bam)], stdout=handle, text=True
            )
        if completed.returncode != 0:
            raise RuntimeError(f"samtools idxstats failed for {output_bam}")
        row["slice_status"] = "ok"
        row["slice_bytes"] = str(output_bam.stat().st_size)
        row["slice_error"] = ""

    add_duplicate_flags(rows)
    write_tsv(rows, TCGA_BAM_MANIFEST_TSV)
    write_json(
        validation_summary(rows),
        TCGA_DATA_DIR / "tcga_brca_rnaseq_bam_manifest_validation.json",
    )
    return TCGA_BAM_MANIFEST_TSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCGA-BRCA ANKRD30A sliced BAMs.")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--slice-region", default=DEFAULT_SLICE_REGION)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-slice", action="store_true")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--workflow-type", action="append")
    parser.add_argument("--all-workflows", action="store_true")
    parser.add_argument("--controlled-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--gdc-api-base", default=DEFAULT_GDC_API_BASE)
    parser.add_argument("--samtools")
    args = parser.parse_args()

    if not args.skip_query:
        query_manifest(args)
    if not args.skip_slice:
        slice_bams(args)


if __name__ == "__main__":
    main()
