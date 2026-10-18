"""Seeded random hex IDs for the synthetic data generators.

`uuid.uuid4()` draws from `os.urandom`, so it ignores every seed in the
pipeline: before this module, transaction/login/device/beneficiary IDs
changed on every run even though all the numeric data was reproducible,
which also changed which transactions the Phase 4 golden set picked.

`SeededIdFactory` keeps IDs uuid4-shaped but derives them from a seed, on
its own RNG stream. It deliberately does not draw from the generators'
numpy `rng`: consuming that stream for IDs would shift every later draw
and change the generated dataset itself, not just its IDs. The string
seed also keeps it independent of Faker's `random.Random`, which the
generators seed with the same integer.

Runtime IDs (case IDs, approval IDs) are intentionally *not* generated
here -- those should stay unpredictable.
"""

from __future__ import annotations

import random
import uuid


class SeededIdFactory:
    def __init__(self, seed: int) -> None:
        # str seeds are hashed with SHA-512 by random.seed, so this is
        # stable across processes (unaffected by PYTHONHASHSEED).
        self._rng = random.Random(f"bankshield-ids-{seed}")

    def hex(self) -> str:
        """32-char hex string, same shape as `uuid.uuid4().hex`."""
        return uuid.UUID(int=self._rng.getrandbits(128), version=4).hex
