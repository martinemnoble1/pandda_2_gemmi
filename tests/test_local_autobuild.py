"""The local sub-block must be the native lattice, exactly.

The point of cutting a native index sub-block rather than a Cartesian box is
that no interpolation happens: a structure translated into the sub-block frame
samples the same numbers it would have sampled in the full cell. These tests
pin that, on a MONOCLINIC cell -- the case where a naive Cartesian box and the
native lattice disagree most.

History: the first implementation cut a 96^3 box at 0.5 A. The reference frame
is at resolution/0.4999 (0.458/0.485/0.484 A on BAZ2B), so no box voxel
coincided with a native one, every value was a trilinear blend, and the
differential_evolution fit settled in a different basin -- moving real builds by
a median of 3.4 A while every synthetic test still passed.
"""

import numpy as np
import gemmi
import pytest

from pandda_gemmi.autobuild.local_grid import (
    native_subblock_frame, subblock_from_sparse, subblock_from_dense,
    _frac_matrix,
)


class _MockMask:
    def __init__(self, indicies):
        self.indicies = indicies


class _MockFrame:
    """Stand-in for DFrame exposing only what the sub-block cut needs."""
    def __init__(self, unit_cell, spacing, indicies):
        self.unit_cell = unit_cell
        self.spacing = spacing
        self.mask = _MockMask(indicies)


def _orth_matrix(cell):
    cols = [cell.orthogonalize(gemmi.Fractional(1, 0, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 1, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 0, 1))]
    return np.array([[c.x, c.y, c.z] for c in cols]).T


def _synthetic(cell_params, nu, nv, nw, fn):
    """A native gemmi FloatGrid filled with fn(cartesian), plus a MockFrame whose
    sparse 'mask' is every grid point."""
    cell = gemmi.UnitCell(*cell_params)
    grid = gemmi.FloatGrid(nu, nv, nw)
    grid.set_unit_cell(cell)
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    M = _orth_matrix(cell)
    fu, fv, fw = np.meshgrid(np.arange(nu) / nu, np.arange(nv) / nv,
                             np.arange(nw) / nw, indexing="ij")
    cart = np.stack([fu, fv, fw], axis=-1) @ M.T
    arr = fn(cart).astype(np.float32)
    np.array(grid, copy=False)[:, :, :] = arr
    idx = np.nonzero(np.ones((nu, nv, nw), dtype=np.int8))
    frame = _MockFrame(cell_params, (nu, nv, nw), idx)
    return grid, frame, arr[idx], cell


MONO = (50.0, 55.0, 60.0, 90.0, 95.0, 90.0)
# Deliberately NOT a round spacing, and different on each axis -- 50/109,
# 55/117, 60/131 -- so a Cartesian box could not coincide with the lattice.
SHAPE = (109, 117, 131)


def _smooth(c):
    return (np.sin(0.25 * c[..., 0]) + np.cos(0.20 * c[..., 1])
            + 0.5 * np.sin(0.15 * c[..., 2]))


def test_subblock_values_are_the_native_values():
    """Every sub-block voxel must BE a native voxel -- same number, not a
    resampled approximation of one."""
    grid, frame, sparse, cell = _synthetic(MONO, *SHAPE, _smooth)
    native = np.array(grid, copy=False)
    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.5, 0.5, 0.5)).tolist())

    lo, shape, sub_cell, origin = native_subblock_frame(frame, centroid, 8.0)
    block = np.array(subblock_from_sparse(frame, sparse, lo, shape, sub_cell),
                     copy=False)

    nu, nv, nw = SHAPE
    expected = native[np.ix_((np.arange(shape[0]) + lo[0]) % nu,
                             (np.arange(shape[1]) + lo[1]) % nv,
                             (np.arange(shape[2]) + lo[2]) % nw)]
    assert np.array_equal(block, expected), "sub-block is not the native block"


def test_subblock_frame_reproduces_native_sampling_exactly():
    """THE property the design rests on: sampling the sub-block at a position
    translated by -origin must equal sampling the full grid at that position.

    Checked at random NON-lattice points, so it tests the geometry (the sub-cell
    lengths and angles), not just the value copy.
    """
    grid, frame, sparse, cell = _synthetic(MONO, *SHAPE, _smooth)
    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.45, 0.5, 0.55)).tolist())

    radius = 8.0
    lo, shape, sub_cell, origin = native_subblock_frame(frame, centroid, radius)
    sub = subblock_from_sparse(frame, sparse, lo, shape, sub_cell)

    rng = np.random.default_rng(0)
    pts = centroid + rng.uniform(-radius * 0.8, radius * 0.8, size=(200, 3))
    full_vals = np.array([grid.interpolate_value(gemmi.Position(*p)) for p in pts])
    sub_vals = np.array([sub.interpolate_value(gemmi.Position(*(p - origin)))
                         for p in pts])

    assert np.allclose(full_vals, sub_vals, atol=1e-5), \
        f"max deviation {np.abs(full_vals - sub_vals).max():.2e}"


def test_subblock_covers_the_requested_radius():
    """The block must contain the whole cube, or the fit can translate a pose
    out of the density it is being scored against."""
    _, frame, sparse, cell = _synthetic(MONO, *SHAPE, _smooth)
    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.5, 0.5, 0.5)).tolist())
    radius = 8.0
    lo, shape, sub_cell, origin = native_subblock_frame(frame, centroid, radius)

    M_sub = _orth_matrix(sub_cell)
    for corner in [(-1, -1, -1), (1, 1, 1), (1, -1, 1), (-1, 1, -1)]:
        p = centroid + radius * np.array(corner) - origin
        frac = np.linalg.solve(M_sub, p)
        assert np.all(frac >= 0) and np.all(frac <= 1), \
            f"corner {corner} falls outside the sub-block at frac {frac}"


def test_dense_channel_resampled_onto_the_same_subblock():
    """The raw xmap is on its own lattice (sample_rate=3 vs resolution/0.4999),
    so the frame's mask indices do not address it -- it must be resampled onto
    the sub-block from its own dense array.

    Regression: cutting it via the frame's sparse indices silently read the
    wrong voxels, corrupting the build CNN's xmap channel and moving poses by
    up to 22 A.
    """
    _, frame, _, cell = _synthetic(MONO, *SHAPE, _smooth)
    raw_grid, _, _, _ = _synthetic(MONO, 60, 66, 72, _smooth)   # coarser lattice
    raw_dense = np.array(raw_grid, copy=False)
    assert raw_dense.shape != SHAPE, "shapes must differ for this test to mean anything"

    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.45, 0.5, 0.55)).tolist())
    lo, shape, sub_cell, origin = native_subblock_frame(frame, centroid, 8.0)
    got = np.array(subblock_from_dense(raw_dense, frame, lo, shape, sub_cell),
                   copy=False)

    # Reference: gemmi interpolating the raw grid at the sub-block voxel
    # positions, i.e. what the full-cell path's CNN would have sampled.
    M_sub = _orth_matrix(sub_cell)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                   axis=-1).reshape(-1, 3) / np.array(shape)
    cart = idx @ M_sub.T + origin
    ref = np.array([raw_grid.interpolate_value(gemmi.Position(*p)) for p in cart])
    ref = ref.reshape(shape)

    assert np.corrcoef(got.ravel(), ref.ravel())[0, 1] > 0.9999
    assert np.abs(got - ref).max() < 1e-4


def test_frac_matrix_inverts_orth():
    """Sanity: F (frac<-cart) is the inverse of the orthogonalisation matrix."""
    cell = gemmi.UnitCell(*MONO)
    assert np.allclose(_frac_matrix(cell) @ _orth_matrix(cell), np.eye(3), atol=1e-6)


def test_aligned_dense_channel_is_exact_not_resampled():
    """With align_to, the second lattice's sub-block is EXACT too.

    The raw xmap is on its own lattice, and resampling it was the last residual
    difference from the full-cell path: it feeds the build CNN's xmap channel,
    which arbitrates between near-tied DE restarts, so a hair of interpolation
    error flipped one build by 0.35 A. Both lattices divide the same cell, so
    snapping the block to their shared grid points makes both exact.
    """
    ref_shape, raw_shape = (60, 80, 60), (50, 72, 54)
    _, frame, _, cell = _synthetic(MONO, *ref_shape, _smooth)
    raw_grid, _, _, _ = _synthetic(MONO, *raw_shape, _smooth)
    raw_dense = np.array(raw_grid, copy=False)

    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.45, 0.5, 0.55)).tolist())
    lo, shape, sub_cell, origin = native_subblock_frame(
        frame, centroid, 6.0, align_to=raw_shape)

    got = np.array(subblock_from_dense(raw_dense, frame, lo, shape, sub_cell),
                   copy=False)

    # The exact block of the RAW lattice covering the same region.
    ratio = np.array(raw_shape) / np.array(ref_shape)
    lo_o = np.round(lo * ratio).astype(int)
    shp_o = tuple(int(v) for v in np.round(np.array(shape) * ratio))
    expected = raw_dense[np.ix_((np.arange(shp_o[0]) + lo_o[0]) % raw_shape[0],
                                (np.arange(shp_o[1]) + lo_o[1]) % raw_shape[1],
                                (np.arange(shp_o[2]) + lo_o[2]) % raw_shape[2])]

    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert np.array_equal(got, expected), "aligned dense channel was resampled, not copied"


def test_aligned_block_still_covers_the_radius():
    """Snapping grows the block; it must still contain the requested cube."""
    ref_shape, raw_shape = (60, 80, 60), (50, 72, 54)
    _, frame, _, cell = _synthetic(MONO, *ref_shape, _smooth)
    centroid = np.array(cell.orthogonalize(gemmi.Fractional(0.5, 0.5, 0.5)).tolist())
    radius = 6.0
    lo, shape, sub_cell, origin = native_subblock_frame(
        frame, centroid, radius, align_to=raw_shape)
    M_sub = _orth_matrix(sub_cell)
    for corner in [(-1, -1, -1), (1, 1, 1), (1, -1, 1), (-1, 1, -1)]:
        frac = np.linalg.solve(M_sub, centroid + radius * np.array(corner) - origin)
        assert np.all(frac >= 0) and np.all(frac <= 1), f"corner {corner} outside"
