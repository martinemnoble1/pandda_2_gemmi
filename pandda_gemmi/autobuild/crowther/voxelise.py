"""Gaussian "calc_fc-lite" stamping of a conformer onto the orthonormal cube.

Lifted from FragVol ``inspect_mr_sh.py`` (``make_gaussian_stamp`` /
``voxelise_gaussian``), with the per-atom weighting graft from FragVol
``fragvol.py`` ``stamp_weighted_gaussians_into_grid`` folded in so the probe can
be Z-weighted (a single-Gaussian resolution-shell low-pass calc_fc) rather than
unit-weighted.

Same orthonormal-P1 invariant as ``rotation.py``: ``origin`` is Cartesian (A),
``spacing`` is an isotropic scalar (A), grid is C-order.

NB: set ``sigma`` from the dataset *resolution*, not the cube spacing. The cube
typically upsamples the native map (PanDDA samples at ~res/2), so a fine cube
spacing must not be mistaken for fine effective resolution.
"""

from __future__ import annotations

import numpy as np

DTYPE_GRID = np.float32


def make_gaussian_stamp(sigma: float, spacing: float):
    """Return (stamp, r_vox): the (2r+1)^3 Gaussian weights centred on the
    stamp's central voxel; r_vox = ceil(3 sigma / spacing)."""
    r_vox = int(np.ceil(3 * sigma / spacing))
    ax = np.arange(-r_vox, r_vox + 1, dtype=DTYPE_GRID) * spacing
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    g = np.exp(-(X * X + Y * Y + Z * Z) / (2 * sigma * sigma)).astype(DTYPE_GRID)
    return g, r_vox


def voxelise_gaussian(coords: np.ndarray, origin: np.ndarray, spacing: float,
                      grid: int, stamp: np.ndarray, r_vox: int,
                      weights: np.ndarray | None = None) -> np.ndarray:
    """Place Gaussian stamps at each atom centre. Returns (grid, grid, grid) fp32.

    ``weights`` (optional, shape (N_atoms,)): per-atom scale, e.g. atomic number
    Z for a calc_fc-like target. If None, every atom contributes an identical
    unit Gaussian (the original FragVol behaviour).

    Atoms outside the cube are skipped by the clipped slice; sub-voxel offsets
    use the precomputed stamp (negligible aliasing at 0.5-1 A spacing).
    """
    n = grid
    occ = np.zeros((n, n, n), dtype=DTYPE_GRID)
    if coords.size == 0:
        return occ
    if weights is None:
        weights = np.ones(coords.shape[0], dtype=DTYPE_GRID)
    else:
        weights = np.asarray(weights, dtype=DTYPE_GRID).reshape(-1)
        if weights.shape[0] != coords.shape[0]:
            raise ValueError(
                f"weights length {weights.shape[0]} != coords rows {coords.shape[0]}")
    rel = (coords - origin[None, :]) / spacing
    for (cx, cy, cz), w in zip(rel, weights):
        ix0, iy0, iz0 = int(np.floor(cx)), int(np.floor(cy)), int(np.floor(cz))
        ix_lo, ix_hi = max(ix0 - r_vox, 0), min(ix0 + r_vox + 1, n)
        iy_lo, iy_hi = max(iy0 - r_vox, 0), min(iy0 + r_vox + 1, n)
        iz_lo, iz_hi = max(iz0 - r_vox, 0), min(iz0 + r_vox + 1, n)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue
        sx_lo = ix_lo - (ix0 - r_vox); sx_hi = sx_lo + (ix_hi - ix_lo)
        sy_lo = iy_lo - (iy0 - r_vox); sy_hi = sy_lo + (iy_hi - iy_lo)
        sz_lo = iz_lo - (iz0 - r_vox); sz_hi = sz_lo + (iz_hi - iz_lo)
        occ[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] += \
            float(w) * stamp[sx_lo:sx_hi, sy_lo:sy_hi, sz_lo:sz_hi]
    return occ
