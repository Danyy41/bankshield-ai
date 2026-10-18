"""Verify that regenerated metrics match the committed ones.

Run after `python scripts/run_all.py`. For every committed
`reports/metrics/*.json`, compares the freshly regenerated file in the
working tree against the version at git HEAD and exits non-zero on any
difference -- i.e. if the pipeline is no longer reproducible, or if a code
or dependency change moved the numbers without the committed reports
being regenerated.

Ignored when comparing:
- the `environment` block (recorded for traceability, not a result)
- wall-clock latency fields in the Phase 4 eval (inherently run-dependent)

Floats are compared with a tight relative tolerance (1e-9) rather than
bit-exactly, so harmless last-ulp noise across CPUs doesn't fail the check
while any real metric change still does. Integers and strings must match
exactly.

Usage:
    python scripts/check_reproducibility.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bankshield import config
from bankshield.reproducibility import ENVIRONMENT_KEY

VOLATILE_KEYS = {ENVIRONMENT_KEY, "latency_ms", "mean_latency_ms"}
REL_TOL = 1e-9


def _committed(path: Path) -> dict | None:
    rel = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _diff(committed, regenerated, where: str) -> list[str]:
    if isinstance(committed, dict) and isinstance(regenerated, dict):
        diffs = []
        for key in sorted(set(committed) | set(regenerated)):
            if key in VOLATILE_KEYS:
                continue
            if key not in committed or key not in regenerated:
                diffs.append(f"{where}.{key}: present in only one version")
                continue
            diffs.extend(_diff(committed[key], regenerated[key], f"{where}.{key}"))
        return diffs
    if isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            return [f"{where}: length {len(committed)} -> {len(regenerated)}"]
        diffs = []
        for i, (a, b) in enumerate(zip(committed, regenerated)):
            diffs.extend(_diff(a, b, f"{where}[{i}]"))
        return diffs
    if isinstance(committed, float) or isinstance(regenerated, float):
        if (
            isinstance(committed, (int, float))
            and isinstance(regenerated, (int, float))
            and math.isclose(committed, regenerated, rel_tol=REL_TOL, abs_tol=1e-12)
        ):
            return []
    elif committed == regenerated:
        return []
    return [f"{where}: committed {committed!r} -> regenerated {regenerated!r}"]


def main() -> int:
    paths = sorted(config.METRICS_DIR.glob("*.json"))
    if not paths:
        print(f"No metrics found in {config.METRICS_DIR} -- run `python scripts/run_all.py` first.")
        return 1

    failures = 0
    for path in paths:
        committed = _committed(path)
        if committed is None:
            print(f"[SKIP] {path.name}: not committed at HEAD")
            continue
        regenerated = json.loads(path.read_text(encoding="utf-8"))
        diffs = _diff(committed, regenerated, path.stem)
        if not diffs:
            print(f"[ OK ] {path.name}")
            continue

        failures += 1
        print(f"[FAIL] {path.name}: {len(diffs)} difference(s)")
        for line in diffs[:10]:
            print(f"         {line}")
        if len(diffs) > 10:
            print(f"         ... and {len(diffs) - 10} more")
        env_then, env_now = committed.get(ENVIRONMENT_KEY), regenerated.get(ENVIRONMENT_KEY)
        if env_then != env_now:
            print(f"         environment differs -- committed: {json.dumps(env_then)}")
            print(f"                                 current: {json.dumps(env_now)}")

    if failures:
        print(
            f"\n{failures} metrics file(s) differ from the committed versions. If this is an "
            "environment mismatch, install the pinned stack:\n"
            "    pip install -r requirements-dev.txt -c requirements-lock.txt\n"
            "If a code or dependency change intentionally moved the numbers, rerun "
            "`python scripts/run_all.py` and commit the regenerated reports/."
        )
        return 1
    print(f"\nAll {len(paths)} metrics files match the committed versions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
