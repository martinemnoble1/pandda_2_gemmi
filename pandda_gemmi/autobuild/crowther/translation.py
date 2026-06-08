"""FFT translation function + clash-penalised Tanimoto scoring.

Lifted from FragVol ``inspect_clash.py`` (``refine_translation_with_clash`` /
``PoseScore``). After the SH rotation function gives a top-K orientation, this
finds the best *translation* by FFT cross-correlation, scored by a
magnitude-robust Tanimoto with a soft protein-clash penalty.

This is the FRF-native replacement for the current code's negative-probe term
(``score_fit_mask_diff_array`` in ``autobuild.inbuilt``): Tanimoto answers "is
the density ligand-shaped", clash answers "does it overlap protein", and the
translation step is the only stage that commits a position.

Same orthonormal-P1 invariant as the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseScore:
    tanimoto_at_best_combined: float
    clash_at_best_combined: float
    combined: float
    best_translation_voxel: tuple  # (i, j, k) of the optimum on the translation grid


def refine_translation_with_clash(
        F_probe: np.ndarray,
        F_target_conj: np.ndarray,
        F_protein_conj: np.ndarray,
        target_self: float,
        probe_self: float,
        lambda_clash: float,
        grid: int,
) -> PoseScore:
    """For a probe at a fixed rotation, search over translation to find the pose
    maximising (Tanimoto - lambda . clash / probe_self).

    Both the target overlap (Tanimoto numerator) and the clash overlap come from
    translation-FFT IFFTs of ``F_probe`` against the respective reference fields.
    Pass ``F_protein_conj`` = zeros and ``lambda_clash`` = 0 to disable clash.

    NB (write-back): ``best_translation_voxel`` is an index on the *cyclic*
    translation grid. A peak past n/2 is a negative shift -- map it via
    ((idx + n//2) % n) - n//2) * spacing in fit.py. This off-by-one wants a
    unit test (see fit.py HOLE 4).
    """
    cc_target = np.fft.irfftn(F_probe * F_target_conj, s=(grid,) * 3)
    cc_protein = np.fft.irfftn(F_probe * F_protein_conj, s=(grid,) * 3)

    union = target_self + probe_self - cc_target
    tani = np.where(union > 0, cc_target / union, 0.0)

    clash = cc_protein / max(probe_self, 1e-12)

    combined = tani - lambda_clash * clash

    idx = np.unravel_index(combined.argmax(), combined.shape)
    return PoseScore(
        tanimoto_at_best_combined=float(tani[idx]),
        clash_at_best_combined=float(clash[idx]),
        combined=float(combined[idx]),
        best_translation_voxel=tuple(int(x) for x in idx),
    )
