"""Correctness tests for the SH-Crowther FRF ligand fit (pandda_gemmi.autobuild.crowther).

Synthetic, self-contained (no checkpoints / no bundle). They pin the two
silent-failure modes flagged as holes during scaffolding:

  - HOLE 4: the translation-FFT wrap/sign convention (plant a probe at a known
    offset, recover it end-to-end through fit_conformer_crowther).
  - HOLE 7: the cube cut-out axis order (a transpose would be silent-wrong).

plus a fast regression of the FRF identity self-test and a rotation-recovery
sanity check.
"""

import numpy as np
import gemmi
import pytest

from pandda_gemmi.autobuild.crowther import rotation as rot, voxelise as vox
from pandda_gemmi.autobuild.crowther.translation import refine_translation_with_clash
from pandda_gemmi.autobuild.crowther.fit import (
    CrowtherConfig, build_precompute, cut_cube, fit_conformer_crowther, _voxel_to_shift,
    sigma_from_resolution, prepare_event_target, fit_conformer_against,
)


# --- synthetic helpers ------------------------------------------------------

# An asymmetric heavy-atom cluster so the Patterson and the axis test are
# non-degenerate (no accidental symmetry to hide a transpose/sign bug).
_COORDS = np.array([
    [0.0, 0.0, 0.0],
    [2.3, 0.0, 0.0],
    [0.0, 3.1, 0.0],
    [0.0, 0.0, 4.2],
    [1.4, 1.1, 0.3],
    [-1.8, 0.6, 2.0],
], dtype=np.float64)


def _gemmi_structure(coords):
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("X")
    res = gemmi.Residue()
    res.name = "LIG"
    res.seqid = gemmi.SeqId(1, " ")
    for i, (x, y, z) in enumerate(coords):
        a = gemmi.Atom()
        a.name = f"C{i}"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(float(x), float(y), float(z))
        res.add_atom(a)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def _density_grid(coords, cell_edge=40.0, n=80, sigma=1.0):
    """A P1 cubic FloatGrid with unit Gaussians stamped at ``coords`` (world A)."""
    grid = gemmi.FloatGrid(n, n, n)
    grid.set_unit_cell(gemmi.UnitCell(cell_edge, cell_edge, cell_edge, 90, 90, 90))
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    sp = cell_edge / n
    origin = np.zeros(3)
    stamp, r_vox = vox.make_gaussian_stamp(sigma, sp)
    arr = vox.voxelise_gaussian(coords, origin, sp, n, stamp, r_vox)
    np.array(grid, copy=False)[:, :, :] = arr
    return grid


def _orth_matrix(cell):
    """Fractional->Cartesian matrix, taken straight from gemmi so it is exactly
    the inverse of the fractionalisation cut_cube's interpolate_values uses."""
    cols = [cell.orthogonalize(gemmi.Fractional(1, 0, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 1, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 0, 1))]
    return np.array([[c.x, c.y, c.z] for c in cols]).T  # columns = basis vectors


def _monoclinic_density_grid(coords_cart, a=40.0, b=40.0, c=40.0, beta=94.7,
                             n=64, sigma=1.0):
    """A non-orthogonal (P 1 2_1 1-like) FloatGrid with Gaussians at Cartesian
    ``coords_cart``. Filled via gemmi's own orthogonalisation so the geometry is
    self-consistent with cut_cube. Returns (grid, M) where M maps frac->Cart."""
    cell = gemmi.UnitCell(a, b, c, 90.0, beta, 90.0)
    M = _orth_matrix(cell)
    grid = gemmi.FloatGrid(n, n, n)
    grid.set_unit_cell(cell)
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    # Cartesian position of every voxel: cart = M @ (u/n, v/n, w/n)
    idx = np.arange(n) / n
    fu, fv, fw = np.meshgrid(idx, idx, idx, indexing="ij")
    frac = np.stack([fu, fv, fw], axis=-1)            # (n,n,n,3)
    cart = frac @ M.T                                 # (n,n,n,3)
    arr = np.zeros((n, n, n), dtype=np.float32)
    two_s2 = 2 * sigma * sigma
    for atom in coords_cart:
        d2 = ((cart - atom) ** 2).sum(axis=-1)
        arr += np.exp(-d2 / two_s2).astype(np.float32)
    np.array(grid, copy=False)[:, :, :] = arr
    return grid, M


# --- HOLE 7: cube cut-out axis order ---------------------------------------

def test_cut_cube_axis_order():
    """cube[i,j,k] must sample the source grid at origin + spacing*(i,j,k).
    A linear field f = x + 10y + 100z is trilinear-exact, so distinct
    per-axis weights catch any axis transpose."""
    cell_edge, n = 40.0, 80
    grid = gemmi.FloatGrid(n, n, n)
    grid.set_unit_cell(gemmi.UnitCell(cell_edge, cell_edge, cell_edge, 90, 90, 90))
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    arr = np.array(grid, copy=False)
    sp_src = cell_edge / n
    ii = np.arange(n) * sp_src
    arr[:, :, :] = (ii[:, None, None] * 1.0
                    + ii[None, :, None] * 10.0
                    + ii[None, None, :] * 100.0).astype(np.float32)

    centroid = np.array([20.0, 20.0, 20.0])
    cube, origin, sp = cut_cube(grid, centroid, n=16, spacing=0.5)

    for (i, j, k) in [(3, 5, 7), (10, 2, 12), (8, 8, 8)]:
        expected = ((origin[0] + sp * i) * 1.0
                    + (origin[1] + sp * j) * 10.0
                    + (origin[2] + sp * k) * 100.0)
        assert cube[i, j, k] == pytest.approx(expected, abs=1e-2), \
            f"axis mismatch at {(i, j, k)}: {cube[i, j, k]} vs {expected}"


# --- FRF identity self-test (fast regression) ------------------------------

def test_frf_identity_self_correlation():
    """SH rotation score at identity == direct radial integral of the Patterson
    (up to SH truncation at L)."""
    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=8, n_r=12, n_rotations=64)
    pre = build_precompute(cfg)
    n, sp = cfg.grid, cfg.spacing
    stamp, r_vox = vox.make_gaussian_stamp(cfg.sigma, sp)
    origin = (-n / 2 * sp * np.ones(3)).astype(np.float32)
    centre = np.zeros(3)
    dens = vox.voxelise_gaussian(_COORDS + (n / 2 * sp), origin, sp, n, stamp, r_vox)
    pat = rot.compute_patterson(dens)
    sph = rot.sample_density_on_spheres(pat, origin, sp, centre, pre.r, pre.theta, pre.phi)
    f = rot.sh_expand_fast(sph, pre.Y_conj)
    X = rot.cross_corr_tensor(f, f, pre.r, pre.dr, cfg.L_max)
    ident = rot.rotation_score(X, 0.0, 0.0, 0.0, cfg.L_max)
    # integration weights aren't stored on the precompute; recompute them
    _, _, _, _, _, w_t, _, w_phi = rot.make_spherical_grid(cfg.L_max, cfg.n_r, n / 2 * sp)
    direct = float(((sph ** 2) * w_t[None, :, None] * w_phi
                    * (pre.r ** 2 * pre.dr)[:, None, None]).sum())
    assert abs(ident - direct) / abs(direct) < 0.05


# --- HOLE 4: translation wrap/sign, end-to-end through fit -----------------

@pytest.mark.parametrize("t_true", [
    (1.5, -1.0, 0.5),
    (-2.0, 0.5, 1.0),
    (3.0, -2.5, 0.0),  # beyond n/2 on no axis at n=32 -> stays a clean lag
    (0.0, 0.0, 0.0),
])
def test_translation_convention(t_true):
    """Pure translation-FFT sign/wrap test, no rotation, no masking. Plant the
    target as the probe shifted by +t_true; _voxel_to_shift(best voxel) must
    return exactly +t_true (the shift to add to the probe). This is the
    deterministic guard for the convention fit_conformer_crowther relies on."""
    n, sp = 32, 0.5
    centre = np.array([n / 2 * sp] * 3)
    origin = np.zeros(3)
    t_true = np.array(t_true)
    stamp, r_vox = vox.make_gaussian_stamp(1.0, sp)
    target = vox.voxelise_gaussian(_COORDS + centre + t_true, origin, sp, n, stamp, r_vox)
    probe = vox.voxelise_gaussian(_COORDS + centre, origin, sp, n, stamp, r_vox)
    Ft_conj = np.fft.rfftn(target).conj()
    Fp = np.fft.rfftn(probe)
    pose = refine_translation_with_clash(
        Fp, Ft_conj, np.zeros_like(Ft_conj),
        float((target * target).sum()), float((probe * probe).sum()), 0.0, n)
    shift = _voxel_to_shift(pose.best_translation_voxel, n, sp)
    assert np.allclose(shift, t_true, atol=1e-6), \
        f"recovered shift {shift} != t_true {t_true}"


def _heavy_coords(structure):
    out = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        p = atom.pos
                        out.append([p.x, p.y, p.z])
    return np.array(out)


# Start orientations: identity and a generic rotation. The whole point of the
# FRF is to reposition an arbitrarily-oriented conformer, so the arbitrary case
# is the load-bearing one -- the conformer goes in mis-oriented and the fit must
# recover BOTH the rotation and the translation onto the offset target.
@pytest.mark.parametrize("euler_start", [
    (0.0, 0.0, 0.0),
    (55.0, -30.0, 80.0),
    (120.0, 70.0, -40.0),
])
def test_fit_repositions_arbitrarily_rotated_model(euler_start):
    """Full fit_conformer_crowther. The target is the canonical conformer at
    centre + t_true; the *input* conformer is the same molecule in an arbitrary
    starting orientation. The fitted heavy atoms must land on the truth coords
    (joint rotation + translation recovery) to seed accuracy.

    RMSD-to-truth (not centroid) is the metric: it is the only thing that tests
    the recovered orientation, since a mean-centred conformer's pose centroid is
    rotation-independent. For a chiral cluster the Patterson handedness alias is
    a mirror image (improper), so the top-K + Tanimoto step rejects it and the
    proper orientation wins."""
    from scipy.spatial.transform import Rotation
    centre = np.array([20.0, 20.0, 20.0])
    t_true = np.array([1.5, -1.0, 0.5])
    truth = _COORDS + centre + t_true
    target_grid = _density_grid(truth)

    R_start = Rotation.from_euler("xyz", euler_start, degrees=True).as_matrix()
    conformer = _gemmi_structure(_COORDS @ R_start.T)  # arbitrary start pose

    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=12, n_r=14,
                    n_rotations=3000, sigma=1.0, top_k=30)
    pre = build_precompute(cfg)

    st, _score, _pc = fit_conformer_crowther(
        centre, conformer, target_grid, pre, ligand_radius=6.0)

    placed = _heavy_coords(st)
    rmsd = np.sqrt(((placed - truth) ** 2).sum(axis=1).mean())
    assert rmsd < 1.5, f"RMSD-to-truth {rmsd:.2f} A (start euler {euler_start})"


# --- rotation recovery sanity ----------------------------------------------

def test_rotation_recovery_sanity():
    """A probe rotated by a known R, fit against the unrotated target: the FRF
    top rotation must give high real-space overlap (>0.7) and beat a random
    rotation. Lenient (SO(3) is sampled, not exhaustive)."""
    from scipy.spatial.transform import Rotation
    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=12, n_r=14, n_rotations=3000)
    pre = build_precompute(cfg)
    n, sp = cfg.grid, cfg.spacing
    centre = np.array([n / 2 * sp] * 3)
    stamp, r_vox = vox.make_gaussian_stamp(cfg.sigma, sp)
    origin = np.zeros(3)

    target = vox.voxelise_gaussian(_COORDS + centre, origin, sp, n, stamp, r_vox)
    target_self = float((target * target).sum())

    # build target SH from its Patterson
    pat_origin = (-n / 2 * sp * np.ones(3)).astype(np.float32)
    t_sph = rot.sample_density_on_spheres(rot.compute_patterson(target),
                                          pat_origin, sp, np.zeros(3),
                                          pre.r, pre.theta, pre.phi)
    f_target = rot.sh_expand_fast(t_sph, pre.Y_conj)

    R_true = Rotation.from_euler("xyz", [40, -25, 70], degrees=True).as_matrix()
    probe_coords = _COORDS @ R_true.T
    probe = vox.voxelise_gaussian(probe_coords + centre, origin, sp, n, stamp, r_vox)
    p_sph = rot.sample_density_on_spheres(rot.compute_patterson(probe),
                                          pat_origin, sp, np.zeros(3),
                                          pre.r, pre.theta, pre.phi)
    f_probe = rot.sh_expand_fast(p_sph, pre.Y_conj)

    X = rot.cross_corr_tensor(f_target, f_probe, pre.r, pre.dr, cfg.L_max)
    scores = rot.score_all_rotations(X, pre.D_batch)
    best = int(np.argmax(scores))

    def overlap(Rm):
        pc = probe_coords @ Rm.T + centre
        pg = vox.voxelise_gaussian(pc, origin, sp, n, stamp, r_vox)
        cc = np.fft.irfftn(np.fft.rfftn(pg) * np.fft.rfftn(target).conj(), s=(n,) * 3)
        return float(cc.max()) / np.sqrt(target_self * float((pg * pg).sum()))

    best_overlap = overlap(pre.Rs_mat[best])
    rand_overlap = overlap(pre.Rs_mat[(best + 137) % cfg.n_rotations])
    assert best_overlap > 0.7, f"best overlap only {best_overlap:.2f}"
    assert best_overlap > rand_overlap


# --- HOLE 6: sigma from resolution ----------------------------------------

def test_sigma_from_resolution():
    """FWHM == resolution mapping, with clamping."""
    assert sigma_from_resolution(2.3548) == pytest.approx(1.0, abs=1e-3)
    assert sigma_from_resolution(0.5) == 0.5      # clamped low
    assert sigma_from_resolution(10.0) == 2.0     # clamped high
    # monotonic in the unclamped band
    assert sigma_from_resolution(1.5) < sigma_from_resolution(2.5)


def test_fit_sigma_override_is_used_not_cached_config():
    """A sigma override must drive voxelisation regardless of the cached
    precompute's config.sigma (the cache is sigma-independent)."""
    centre = np.array([20.0, 20.0, 20.0])
    t_true = np.array([1.5, -1.0, 0.5])
    truth = _COORDS + centre + t_true
    target_grid = _density_grid(truth)
    conformer = _gemmi_structure(_COORDS)

    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=12, n_r=14,
                    n_rotations=2000, sigma=1.0, top_k=30)
    pre = build_precompute(cfg)

    # override with a resolution-derived sigma; pose recovery must still hold
    sigma = sigma_from_resolution(2.0)
    assert sigma != cfg.sigma
    st, _score, _c = fit_conformer_crowther(
        centre, conformer, target_grid, pre, ligand_radius=6.0, sigma=sigma)
    rmsd = np.sqrt(((_heavy_coords(st) - truth) ** 2).sum(axis=1).mean())
    assert rmsd < 1.5, f"RMSD {rmsd:.2f} A with sigma override {sigma:.2f}"


# --- non-orthogonal source geometry (the real data is P 1 2_1 1) -----------

def test_fit_on_monoclinic_source_grid():
    """cut_cube makes NO spacegroup/geometry assumption about the source map: it
    samples Cartesian positions and lets gemmi fractionalise through the source
    grid's own cell (exactly as the existing 32^3 CNN SampleFrame box does). On a
    monoclinic source (P 1 2_1 1, beta=94.7) an arbitrarily-rotated conformer
    must still be repositioned onto the orthonormal-cube-sampled density to seed
    accuracy."""
    from scipy.spatial.transform import Rotation
    a = b = c = 40.0
    beta = 94.7
    cell = gemmi.UnitCell(a, b, c, 90.0, beta, 90.0)
    centre = _orth_matrix(cell) @ np.array([0.5, 0.5, 0.5])  # Cartesian, in-cell
    t_true = np.array([1.5, -1.0, 0.5])
    truth = _COORDS + centre + t_true
    grid, _M = _monoclinic_density_grid(truth, a, b, c, beta, n=64, sigma=1.0)

    R_start = Rotation.from_euler("xyz", [55, -30, 80], degrees=True).as_matrix()
    conformer = _gemmi_structure(_COORDS @ R_start.T)

    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=12, n_r=14,
                    n_rotations=2000, sigma=1.0, top_k=30)
    pre = build_precompute(cfg)
    st, _s, _c = fit_conformer_crowther(centre, conformer, grid, pre, ligand_radius=6.0)
    rmsd = np.sqrt(((_heavy_coords(st) - truth) ** 2).sum(axis=1).mean())
    assert rmsd < 1.5, f"monoclinic-source RMSD {rmsd:.2f} A"


# --- per-event target prep reused across conformers ------------------------

def test_event_target_prepared_once_fits_many_conformers():
    """The event/z source is a per-event object: prepare_event_target() derives
    the cut + Patterson + SH expansion ONCE, then fit_conformer_against() places
    each conformer with only its own (local, in-cube) density. Several
    differently-oriented conformers must all be recovered from the single
    prepared target -- the structural fix for the per-conformer regeneration."""
    from scipy.spatial.transform import Rotation
    centre = np.array([20.0, 20.0, 20.0])
    t_true = np.array([1.5, -1.0, 0.5])
    truth = _COORDS + centre + t_true
    target_grid = _density_grid(truth)

    cfg = CrowtherConfig(grid=32, spacing=0.5, L_max=12, n_r=14,
                    n_rotations=2000, sigma=1.0, top_k=30)
    pre = build_precompute(cfg)

    # ONE target preparation for the event...
    event_target = prepare_event_target(
        target_grid, centre, pre, ligand_radius=6.0)

    # ...reused across several conformer orientations.
    for euler in [(0.0, 0.0, 0.0), (55.0, -30.0, 80.0), (200.0, 15.0, -95.0)]:
        R = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        conformer = _gemmi_structure(_COORDS @ R.T)
        st, _s, _c = fit_conformer_against(event_target, conformer, pre)
        rmsd = np.sqrt(((_heavy_coords(st) - truth) ** 2).sum(axis=1).mean())
        assert rmsd < 1.5, f"reuse RMSD {rmsd:.2f} A (euler {euler})"
