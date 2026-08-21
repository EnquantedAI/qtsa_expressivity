"""Public interface for the matched entanglement-topology study.

The stable entry point for this package is the frozen reproducible study.  The
remaining modules are kept as focused diagnostics and development helpers.
"""

from .reproducible_study import (
    DEFAULT_REPRODUCIBLE_PRESET,
    REPRODUCIBLE_STUDY_NAME,
    TopologyStudyPreset,
    preset_manifest,
    run_reproducible_topology_study,
    save_reproducible_topology_study,
)

__all__ = [
    "DEFAULT_REPRODUCIBLE_PRESET",
    "REPRODUCIBLE_STUDY_NAME",
    "TopologyStudyPreset",
    "preset_manifest",
    "run_reproducible_topology_study",
    "save_reproducible_topology_study",
]
