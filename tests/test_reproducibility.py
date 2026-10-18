import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from bankshield.ids import SeededIdFactory
from bankshield.reproducibility import ENVIRONMENT_KEY, TRACKED_PACKAGES, environment_info, with_environment

ROOT = Path(__file__).resolve().parents[1]

# Small Phase 1 -> 2 -> 3 generation, hashed. Run in a subprocess so each
# run gets its own PYTHONHASHSEED: str hashing (and so set iteration
# order) is randomized per process, which is exactly the kind of
# nondeterminism an in-process comparison can't see.
_GENERATE_AND_HASH = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from bankshield.data_generation import generate_transactions
from bankshield.auth_generation import generate_login_events
from bankshield.graph_generation import generate_graph_relationships
txns = generate_transactions(n_transactions=1500, n_customers=200, seed=5)
logins = generate_login_events(txns, seed=6)
graph = generate_graph_relationships(txns, seed=7)
h = hashlib.sha256()
for df in (txns, logins, graph):
    h.update(df.to_csv(index=False).encode())
print(h.hexdigest())
"""


def _generate_hash(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", _GENERATE_AND_HASH, str(ROOT / "src")],
        env=env, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_generated_data_is_byte_identical_across_processes_and_hash_seeds():
    assert _generate_hash("1") == _generate_hash("2")


def test_seeded_ids_are_deterministic_and_uuid4_shaped():
    a, b = SeededIdFactory(42), SeededIdFactory(42)
    ids_a = [a.hex() for _ in range(100)]
    assert ids_a == [b.hex() for _ in range(100)]
    assert len(set(ids_a)) == 100
    assert all(len(i) == 32 and i[12] == "4" for i in ids_a)  # uuid4 version nibble


def test_seeded_ids_differ_by_seed():
    assert SeededIdFactory(1).hex() != SeededIdFactory(2).hex()


def test_environment_info_records_tracked_packages():
    env = environment_info()
    assert env["python"].count(".") == 2
    assert set(env["packages"]) == set(TRACKED_PACKAGES)
    assert all(env["packages"][name] for name in ("numpy", "pandas", "scikit-learn", "xgboost"))


def test_with_environment_does_not_mutate_metrics():
    metrics = {"f1": 0.5}
    out = with_environment(metrics)
    assert metrics == {"f1": 0.5}
    assert out["f1"] == 0.5 and ENVIRONMENT_KEY in out


def _load_check_script():
    spec = importlib.util.spec_from_file_location("check_reproducibility", ROOT / "scripts" / "check_reproducibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_reproducibility_ignores_environment_and_latency_only():
    check = _load_check_script()
    committed = {"f1": 0.25, "mean_latency_ms": 10.0, ENVIRONMENT_KEY: {"python": "3.11.0"},
                 "rows": [{"latency_ms": 1.0, "tool_calls": 4}]}
    same = {"f1": 0.25 * (1 + 1e-12), "mean_latency_ms": 99.0, ENVIRONMENT_KEY: {"python": "3.12.0"},
            "rows": [{"latency_ms": 7.0, "tool_calls": 4}]}
    assert check._diff(committed, same, "m") == []

    moved = {**same, "f1": 0.26}
    assert check._diff(committed, moved, "m") == ["m.f1: committed 0.25 -> regenerated 0.26"]
    assert check._diff(committed, {**same, "rows": [{"latency_ms": 7.0, "tool_calls": 5}]}, "m")
    assert check._diff(committed, {k: v for k, v in same.items() if k != "f1"}, "m")
