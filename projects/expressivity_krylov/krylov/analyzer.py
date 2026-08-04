"""
High-level interface for Krylov-based expressivity analysis.

This module coordinates

    circuit prefixes
        ↓
    Lanczos iteration
        ↓
    Krylov metrics

while caching intermediate results so that increasing-depth analyses
reuse as much computation as possible.

The actual implementations of the individual algorithms live in

    circuits.py
    lanczos.py
    metrics.py
    operators.py

Only this file should be imported by notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pennylane as qml
from pennylane import numpy as np

# ---------------------------------------------------------------------
# Placeholder imports
# ---------------------------------------------------------------------

from .circuits import CircuitPrefix
from .lanczos import LanczosIterator
from .metrics import (
    krylov_expressivity,
    spread_complexity,
    observability,
)

from .operators import build_liouvillian

# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------


@dataclass(slots=True)
class DepthResult:
    """Results for one network depth."""

    depth: int

    state: Any | None = None

    lanczos: Any | None = None

    expressivity: float | None = None

    spread: float | None = None

    observability: float | None = None


@dataclass(slots=True)
class DepthSweepResult:
    """Results for all depths."""

    results: list[DepthResult] = field(default_factory=list)

    @property
    def depths(self):
        return [r.depth for r in self.results]

    @property
    def expressivities(self):
        return [r.expressivity for r in self.results]

    @property
    def spreads(self):
        return [r.spread for r in self.results]

    @property
    def observabilities(self):
        return [r.observability for r in self.results]


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------


class KrylovAnalyzer:
    """
    Performs Krylov-based analysis of a parameterized quantum circuit.

    Parameters
    ----------
    model
        PennyLane QNode.

    weights
        Weight tensor.

    n_layers
        Number of trainable layers.

    observable
        Observable whose operator growth is studied.

    reference_state
        Optional initial state.

    Notes
    -----
    The analyzer caches all intermediate computations so repeated
    requests for deeper prefixes do not recompute shallower ones.
    """

    def __init__(
        self,
        model,
        weights,
        n_layers: int,
        observable=None,
        reference_state=None,
    ):

        self.model = model
        self.weights = weights
        self.n_layers = n_layers

        self.observable = (
            observable
            if observable is not None
            else qml.PauliZ(0)
        )

        self.reference_state = reference_state

        # Helper object responsible for executing prefixes
        self.prefix = CircuitPrefix(
            model=model,
            weights=weights,
        )

        # --------------------------------------------------
        # caches
        # --------------------------------------------------

        self._state_cache = {}

        self._lanczos_cache = {}

        self._result_cache = {}

    # -----------------------------------------------------------------

    def compute_depth(
        self,
        depth: int,
    ) -> DepthResult:
        """
        Compute Krylov metrics for a circuit consisting of the first
        `depth` trainable layers.
        """

        if depth < 1:
            raise ValueError("depth must be positive")

        if depth > self.n_layers:
            raise ValueError("depth exceeds number of layers")

        if depth in self._result_cache:
            return self._result_cache[depth]

        # --------------------------------------------------
        # Execute circuit prefix
        # --------------------------------------------------

        if depth in self._state_cache:

            state = self._state_cache[depth]

        else:

            state = self.prefix.run(depth)

            self._state_cache[depth] = state

        # --------------------------------------------------
        # Build Liouvillian
        # --------------------------------------------------

        liouvillian = build_liouvillian(
            state=state,
            observable=self.observable,
        )

        # --------------------------------------------------
        # Lanczos
        # --------------------------------------------------

        lanczos = LanczosIterator(liouvillian)

        self._lanczos_cache[depth] = lanczos

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        result = DepthResult(
            depth=depth,
            state=state,
            lanczos=lanczos,
            expressivity=krylov_expressivity(lanczos),
            spread=spread_complexity(lanczos),
            observability=observability(lanczos),
        )

        self._result_cache[depth] = result

        return result

    # -----------------------------------------------------------------

    def compute_all_depths(self) -> DepthSweepResult:
        """
        Compute Krylov metrics for every prefix

            1, 2, ..., N.
        """

        sweep = DepthSweepResult()

        for depth in range(1, self.n_layers + 1):
            sweep.results.append(
                self.compute_depth(depth)
            )

        return sweep

    # -----------------------------------------------------------------

    def clear_cache(self):
        """Discard all cached intermediate results."""

        self._state_cache.clear()
        self._lanczos_cache.clear()
        self._result_cache.clear()
