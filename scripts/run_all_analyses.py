"""Run the tracked analysis cache workflows in their documented Conda environments."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess

from scripts.paper_utils import PROJECT_ROOT

WORKFLOWS = {
    "splisosm_atera_brca": (
        "sonic_paper_splisosm",
        "scripts.splisosm_atera_brca.workflow",
    ),
    "splisosm_visiumhd_cbs": (
        "sonic_paper_splisosm",
        "scripts.splisosm_visiumhd_cbs.workflow",
    ),
    "qtest_benchmark": ("sonic_paper", "scripts.qtest_benchmark.workflow"),
    "comparator_benchmark": ("sonic_paper", "scripts.comparator_benchmark.workflow"),
    "glioma_idh": ("sonic_paper", "scripts.glioma_idh.workflow"),
    "embryo_xspecies": ("sonic_paper", "scripts.embryo_xspecies.workflow"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analyses",
        nargs="*",
        choices=tuple(WORKFLOWS),
        metavar="ANALYSIS",
        help="Workflows to run; omit to run all tracked analyses.",
    )
    parser.add_argument("--force", action="store_true", help="Pass --force to each workflow.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running any workflow.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conda = shutil.which("conda")
    if conda is None and not args.dry_run:
        raise SystemExit("conda is not available on PATH")
    conda = conda or "conda"

    analyses = args.analyses or list(WORKFLOWS)
    for analysis in analyses:
        environment, module = WORKFLOWS[analysis]
        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "python",
            "-m",
            module,
        ]
        if args.force:
            command.append("--force")
        print(f"[{analysis}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
