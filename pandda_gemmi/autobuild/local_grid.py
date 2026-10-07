"""Local unmask: build a small density grid around an event WITHOUT allocating
the full unit cell.

PanDDA stores per-event/model maps sparsely (``reference_frame.mask.indicies``
is a 3-tuple ``(U,V,W)`` of native-grid indices, ``sparse.data[i]`` the value at
``(U[i],V[i],W[i])`` on the native ``(nu,nv,nw)`` P1 grid).
``reference_frame.unmask`` densifies the WHOLE cell, and autobuild does that
five or six times per (model x event x conformer) task.

The grid built here is an **exact sub-block of the native lattice**: native
indices ``lo:lo+shape``, values copied verbatim, in a cell whose lengths are
scaled by ``shape/spacing`` and whose angles are the native ones. That choice is
what makes the local path equivalent to the full-cell one rather than merely
similar:

    the orthogonalisation matrix of the sub-cell is ``M_sub = M @ diag(d/n)``,
    so for a voxel at sub-block index ``p`` (native index ``lo + p``)

        M_sub @ (p/d) = M @ ((lo + p)/n) - M @ (lo/n) = cart_native - origin

    i.e. a structure translated by ``-origin`` samples the sub-block at exactly
    the positions it would have sampled in the full grid, and the values there
    are the same numbers. No interpolation, so no resampling error.

A Cartesian cube at some round spacing (the obvious first design) does NOT have
this property: unless the box spacing divides the native spacing, every voxel is
a trilinear blend, the density-fit objective is subtly different, and the
differential_evolution search settles in a different basin. Measured on BAZ2B,
that moved real builds by a median of 3.4 A.

Scoring (CNN, RSCC, masks, BDC) is translation-invariant, so working in the
sub-block frame changes nothing but the memory.
"""

from __future__ import annotations

import itertools

import numpy as np
import gemmi


def _frac_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    """3x3 fractionalisation matrix F such that frac = F @ cartesian. Taken from
    gemmi so it matches its cell geometry exactly (handles non-orthogonal cells)."""
    cols = [cell.fractionalize(gemmi.Position(1, 0, 0)),
            cell.fractionalize(gemmi.Position(0, 1, 0)),
            cell.fractionalize(gemmi.Position(0, 0, 1))]
    return np.array([[c.x, c.y, c.z] for c in cols]).T


def _common_lattice_step(n: np.ndarray, other) -> np.ndarray:
    """Index period on which lattice ``n`` and lattice ``other`` (both spanning
    the same cell) share grid points: ``n / gcd(n, other)``.

    With n=(180,200,120) and other=(150,180,108) that is (6,10,10). Snapping the
    sub-block corner AND extent to this lets the second lattice be cut as an
    EXACT sub-block too, sharing a Cartesian origin with the first, instead of
    being resampled onto it.
    """
    if other is None:
        return np.ones(3, dtype=np.int64)
    n = n.astype(np.int64)
    other = np.asarray(other, dtype=np.int64)
    return n // np.gcd(n, other)


def native_subblock_frame(reference_frame, centroid, radius: float,
                          align_to=None):
    """Native-index sub-block covering the Cartesian cube of half-width
    ``radius`` about ``centroid``.

    Returns ``(lo, shape, sub_cell, origin)``: the native index of the block
    corner, its dimensions, the gemmi cell to give the sub-block grid, and the
    native Cartesian position of its (0,0,0) voxel. Every channel for one event
    must be cut on this same frame, or the grids handed to the scorer are
    mutually offset.
    """
    cell = gemmi.UnitCell(*reference_frame.unit_cell)
    n = np.asarray(reference_frame.spacing, dtype=np.float64)

    # Fractional -> native-index coords of the cube's 8 corners; the sub-block
    # is their bounding box (+1 voxel margin, so interpolation at the cube face
    # still has neighbours on both sides).
    offs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    corners = np.asarray(centroid, dtype=np.float64)[None, :] + radius * offs
    gi = (corners @ _frac_matrix(cell).T) * n[None, :]
    lo = np.floor(gi.min(0)).astype(np.int64) - 1
    hi = np.ceil(gi.max(0)).astype(np.int64) + 2
    # Snap corner DOWN and extent UP onto lattice points shared with
    # `align_to`, so that lattice's sub-block has the same Cartesian origin and
    # a whole number of its voxels -- i.e. it can be cut exactly too.
    step = _common_lattice_step(n, align_to)
    lo = (lo // step) * step
    hi = -((-hi) // step) * step          # ceil-divide, staying integral
    shape = tuple(int(v) for v in (hi - lo))

    # Same angles, lengths scaled by the fraction of the cell the block spans.
    sub_cell = gemmi.UnitCell(
        cell.a * shape[0] / n[0], cell.b * shape[1] / n[1], cell.c * shape[2] / n[2],
        cell.alpha, cell.beta, cell.gamma)

    o = cell.orthogonalize(gemmi.Fractional(*(lo / n)))
    return lo, shape, sub_cell, np.array([o.x, o.y, o.z], dtype=np.float64)


def _as_grid(values: np.ndarray, sub_cell: gemmi.UnitCell) -> gemmi.FloatGrid:
    grid = gemmi.FloatGrid(*values.shape)
    grid.set_unit_cell(sub_cell)
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    np.array(grid, copy=False)[:, :, :] = values
    return grid


def subblock_from_sparse(reference_frame, sparse_data, lo, shape,
                         sub_cell) -> gemmi.FloatGrid:
    """Copy the sparse values whose native index falls in the sub-block. Exact:
    the values are the native ones, at their native positions.

    Points outside the mask are 0, which is what ``unmask`` puts there too.
    """
    nu, nv, nw = reference_frame.spacing
    U, V, W = reference_frame.mask.indicies
    data = np.asarray(sparse_data, dtype=np.float32)

    # A native point recurs every n indices, so fold into the block by modulo.
    uu = (U.astype(np.int64) - lo[0]) % nu
    vv = (V.astype(np.int64) - lo[1]) % nv
    ww = (W.astype(np.int64) - lo[2]) % nw
    sel = (uu < shape[0]) & (vv < shape[1]) & (ww < shape[2])

    block = np.zeros(shape, dtype=np.float32)
    block[uu[sel], vv[sel], ww[sel]] = data[sel]
    return _as_grid(block, sub_cell)


def _trilinear_periodic(arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Trilinear sample of a full-cell dense array at fractional grid ``coords``
    (N,3), wrapping periodically -- the array spans the whole cell, so an index
    past its end is the same density one cell over."""
    n0, n1, n2 = arr.shape
    i0 = np.floor(coords).astype(np.int64)
    f = coords - i0
    out = np.zeros(coords.shape[0], dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                wt = (np.where(di, f[:, 0], 1 - f[:, 0]) *
                      np.where(dj, f[:, 1], 1 - f[:, 1]) *
                      np.where(dk, f[:, 2], 1 - f[:, 2]))
                out += wt * arr[(i0[:, 0] + di) % n0,
                                (i0[:, 1] + dj) % n1,
                                (i0[:, 2] + dk) % n2]
    return out


def subblock_from_dense(dense_array, reference_frame, lo, shape,
                        sub_cell) -> gemmi.FloatGrid:
    """Sample a dense full-cell array that is on its OWN lattice onto the
    sub-block's voxel positions.

    Needed for the raw xmap: it is sampled at ``sample_rate=3`` while the
    reference frame uses ``resolution/0.4999``, so the two grids have different
    shapes and the frame's mask indices do not address this array at all (the
    ``raw_xmap_sparse`` that process_dataset builds is mis-indexed for exactly
    this reason, which is why the full-cell path ignores it and rebuilds from
    the dense array). Only the unit cell is shared -- enough, because both
    lattices span it, so sub-block voxel ``p`` sits at fractional ``(lo+p)/n``
    in either.

    This one channel is interpolated; the density-fit objective is not.
    """
    arr = np.asarray(dense_array, dtype=np.float32)
    n = np.asarray(reference_frame.spacing, dtype=np.float64)
    m = np.asarray(arr.shape, dtype=np.float64)

    # If this lattice's grid points coincide with the sub-block corner and its
    # voxels divide evenly, take the exact block -- no interpolation at all.
    ratio = m / n
    lo_other = lo * ratio
    if (np.allclose(lo_other, np.round(lo_other)) and
            np.allclose(np.asarray(shape) * ratio, np.round(np.asarray(shape) * ratio))):
        lo_o = np.round(lo_other).astype(np.int64)
        shp_o = tuple(int(v) for v in np.round(np.asarray(shape) * ratio))
        m_i = arr.shape
        block = arr[np.ix_((np.arange(shp_o[0]) + lo_o[0]) % m_i[0],
                           (np.arange(shp_o[1]) + lo_o[1]) % m_i[1],
                           (np.arange(shp_o[2]) + lo_o[2]) % m_i[2])]
        sub_cell_o = gemmi.UnitCell(
            sub_cell.a, sub_cell.b, sub_cell.c,
            sub_cell.alpha, sub_cell.beta, sub_cell.gamma)
        return _as_grid(np.ascontiguousarray(block), sub_cell_o)

    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                   axis=-1).reshape(-1, 3).astype(np.float64)
    gi = ((idx + lo[None, :]) / n[None, :]) * m[None, :]
    vals = _trilinear_periodic(arr, gi).reshape(shape).astype(np.float32)
    return _as_grid(vals, sub_cell)
