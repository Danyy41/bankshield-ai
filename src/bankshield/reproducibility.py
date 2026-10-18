"""Record the software environment that produced a set of metrics.

The pipeline is seeded end to end, so on a fixed environment it is
byte-for-byte reproducible -- but the trained models' numbers still move
across library versions (XGBoost/scikit-learn/numpy/pandas releases change
tree construction, solvers, and float handling). Writing the versions
next to the metrics makes every committed number traceable to the exact
stack that produced it; `requirements-lock.txt` pins that stack, and
`scripts/check_reproducibility.py` verifies regenerated metrics against
the committed ones.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

# Distributions whose versions can change pipeline outputs.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
    "Faker",
)

ENVIRONMENT_KEY = "environment"


def environment_info() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": packages,
    }


def with_environment(metrics: dict) -> dict:
    """Return a copy of `metrics` with the current environment attached."""
    return {**metrics, ENVIRONMENT_KEY: environment_info()}


if __name__ == "__main__":  # pragma: no cover - convenience for humans
    import json

    json.dump(environment_info(), sys.stdout, indent=2)
    print()
