"""PanDDA2 adapter for the SH-Crowther FRF ligand fit.

This is the only crystallography-aware file in the package. It:
  1. cuts a power-of-2, voxel-snapped, orthonormal Cartesian cube from a PanDDA2
     event/z map about the event centroid (the orthonormal-P1 guardrail for the
     verbatim core);
  2. preprocesses the experimental target (soft mask + protein-zeroing +
     mean-subtract) before the Patterson;
  3. drives rotation (Patterson FRF) -> top-K -> clash-penalised translation;
  4. writes the winning pose back into the dataset's native Cartesian frame.

Intended drop-in: behind a flag in ``autobuild.inbuilt.score_conformer``,
returning the same 4-tuple ``(optimized_structure, score, centroid, arr)`` so
the downstream CNN/bdc/signal path is untouched.

The expensive precompute (Y_conj, Wigner-D batch, shell grid, SO(3) set) is
dataset-independent: build a single ``CrowtherPrecompute`` at PanDDA2 startup
(alongside get_scoring_models) and broadcast it via processor.put, exactly like
reference_frame_ref. Per-conformer cost is then just the cube cut, the stamp,
two shell samplings, the cross-correlation tensor, score_all_rotations, and the
top-K translation FFTs.

Sections marked ``# HOLE`` are the decisions still to be made; everything else
is wired against the verbatim core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gemmi
from scipy.spatial.transform import Rotation

from . import rotation as rot
from . import voxelise as vox
from .translation import refine_translation_with_clash


# ---------------------------------------------------------------------------
# Dataset-independent precompute (build once, broadcast)
# ---------------------------------------------------------------------------

@dataclass
class CrowtherConfig:
    """FRF fit parameters.

    Memory note: the dominant resident allocation is the Wigner-D batch
    (``n_rotations`` x sum_l (2l+1)^2 complex64), per-worker and shared across all
    conformers -- ~15 MB at the L=8/N=2000 default (261.8 MB at the old L=16/N=5000,
    2.3 MB at L=8/N=300). NOT freed per conformer (it is the amortisation), so
    ``n_rotations`` and ``L_max`` are the per-worker memory levers; per-fit
    transients (~31 MB at grid=64) are freed on return. CrowtherConfig.lean() drops
    N to 300. The per-fit peak scales with grid^3 (cut to grid=32 = 8x less)."""
    grid: int = 64               # power of 2 (radix-2 FFT)
    spacing: float = 0.5         # A, isotropic
    L_max: int = 8               # SH band limit; L-sweep showed L=8==L=16 on
                                 # real data (FRF<->brute 0.00 at all L) -> lean
    n_r: int = 14                # radial shells
    n_rotations: int = 2000      # SO(3) sample count (validated; denser doesn't
                                 # help direct-density, can pick a symmetry alias)
    sigma: float = 1.0           # probe Gaussian width; set from dataset res
    top_k: int = 30              # orientations carried to the translation step
    lambda_clash: float = 0.0    # HOLE 3: clash weight (0 disables)
    headroom: float = 2.5        # Patterson clean-box: cube edge >= headroom x span
    rotation_seed: int = 42
    # Crowther-Blow integration radius for the SH shells. The rotation function
    # must read the ligand's intramolecular (self) vectors -- which extend only
    # to the molecular diameter -- NOT the full box half-width, or r^2-weighted
    # outer shells of noise/cross-vectors swamp the signal on real density.
    # None -> box half-width (legacy); set to ~ligand extent + margin.
    r_max_shell: float | None = 8.0
    # Rotation function: direct density overlap about the (accurate) event
    # centroid by default. The Patterson route squares weak difference-density
    # noise and was shown to diverge from exhaustive brute on real event maps
    # (FRF<->brute up to 4 A) whereas direct density reproduces brute exactly
    # (FRF<->brute 0.00). Patterson kept as opt-in for untrustworthy centroids.
    use_patterson: bool = False

    @classmethod
    def lean(cls, **overrides):
        """Memory-lean preset: ~2 MB D_batch vs ~262 MB. Validate recall on the
        Mac1 set before trusting it as default (FINDINGS sec.6 supports it)."""
        base = dict(L_max=8, n_r=14, n_rotations=300, top_k=30)
        base.update(overrides)
        return cls(**base)


@dataclass
class CrowtherPrecompute:
    config: CrowtherConfig
    r: np.ndarray
    dr: float
    theta: np.ndarray
    phi: np.ndarray
    Y_conj: np.ndarray
    euler: np.ndarray            # (N, 3) ZYZ
    Rs_mat: np.ndarray           # (N, 3, 3)
    D_batch: list                # list of (N, 2l+1, 2l+1)


def sigma_from_resolution(resolution: float,
                          lo: float = 0.5, hi: float = 2.0) -> float:
    """Probe Gaussian width for a single-Gaussian (calc_fc-lite) atom at a given
    map resolution. Chosen so the Gaussian's FWHM equals d_min:
        FWHM = 2*sqrt(2 ln 2) * sigma = resolution  ->  sigma = resolution/2.355
    i.e. the probe blur matches the finest feature the data resolves. Clamped to
    [lo, hi]. Initial mapping (HOLE 6) -- tune against the Mac1 A/B."""
    return float(min(hi, max(lo, resolution / 2.3548)))


_PRECOMPUTE_CACHE: dict = {}


def get_precompute(config: CrowtherConfig) -> CrowtherPrecompute:
    """Cached ``build_precompute``. The tables depend only on these fields, not
    on any dataset, so one build per worker process is reused across every
    (event, conformer) it handles. (A proper integration would build once at
    startup and broadcast via processor.put; this module-level cache is the
    surgical equivalent for the behind-a-flag A/B.)"""
    key = (config.grid, config.spacing, config.L_max, config.n_r,
           config.n_rotations, config.rotation_seed, config.r_max_shell)
    pre = _PRECOMPUTE_CACHE.get(key)
    if pre is None:
        pre = build_precompute(config)
        _PRECOMPUTE_CACHE[key] = pre
    return pre


def build_precompute(config: CrowtherConfig) -> CrowtherPrecompute:
    """Build the dataset-independent tables. Call once at startup.

    r_max defaults to the cube half-width; HOLE 5 (in make_spherical_grid)
    suggests tightening it to ~ligand diameter and dropping the origin shell to
    focus the FRF on intramolecular vectors.
    """
    r_max = config.r_max_shell or (config.grid / 2.0 * config.spacing)
    r, dr, theta, _cos_t, _sin_t, w_t, phi, w_phi = rot.make_spherical_grid(
        config.L_max, config.n_r, r_max)
    Y_conj = rot.precompute_Y_conj(theta, phi, w_t, w_phi, config.L_max)
    rng = Rotation.random(config.n_rotations, random_state=config.rotation_seed)
    euler = rng.as_euler("ZYZ", degrees=False)
    Rs_mat = rng.as_matrix().astype(np.float32)
    D_batch = rot.precompute_D_batch(euler, config.L_max)
    return CrowtherPrecompute(config=config, r=r, dr=dr, theta=theta, phi=phi,
                         Y_conj=Y_conj, euler=euler, Rs_mat=Rs_mat,
                         D_batch=D_batch)


# ---------------------------------------------------------------------------
# Cube cut-out (the orthonormal-P1 guardrail)
# ---------------------------------------------------------------------------

def cut_cube(grid: gemmi.FloatGrid, centroid, n: int, spacing: float):
    """Resample ``grid`` into an n^3 orthonormal Cartesian cube centred on
    ``centroid``. Returns (cube_array, origin, spacing).

    origin is voxel-snapped (fragvol.compute_envelope_frame trick) so the cut
    lands on an exact grid multiple and no half-voxel shift creeps in. The
    transform matrix is spacing*I -> the cube is guaranteed orthonormal and
    isotropic, which is the invariant the verbatim core relies on.

    interpolate_values samples by Cartesian position and gemmi fractionalises
    through ``grid``'s (possibly non-orthogonal native) cell internally, so this
    is correct even when the source map is monoclinic/triclinic.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    half = (n / 2.0) * spacing
    origin = np.round((centroid - half) / spacing) * spacing  # voxel-snapped

    transform = gemmi.Transform()
    transform.mat.fromlist((np.eye(3) * spacing).tolist())
    transform.vec.fromlist(origin.tolist())

    cube = np.zeros((n, n, n), dtype=np.float32)
    grid.interpolate_values(cube, transform)
    # HOLE 7 (axis order): verify gemmi fills cube[i,j,k] in the same C-order
    # the core indexes. A transpose here is silent-wrong; assert against a
    # known asymmetric test density in the unit test.
    return cube, origin.astype(np.float32), float(spacing)


# ---------------------------------------------------------------------------
# Target preprocessing (experimental density -> clean Patterson input)
# ---------------------------------------------------------------------------

def gaussian_lowpass(cube: np.ndarray, spacing: float, sigma: float) -> np.ndarray:
    """Convolve a cube with a Gaussian of width ``sigma`` (A) via Fourier space:
    H(s) = exp(-2 pi^2 sigma^2 |s|^2). Suppresses the noise-dominated high-
    frequency shells of an experimental difference-density cube so its Patterson
    (which squares noise) and the matched-filter correlation are cleaner, and so
    its bandwidth matches the single-Gaussian probe. (This is the cheap half of a
    Wiener filter -- noise-band suppression without an explicit S/N estimate.)"""
    n = cube.shape[0]
    fx = np.fft.fftfreq(n, d=spacing)
    fz = np.fft.rfftfreq(n, d=spacing)
    s2 = (fx[:, None, None] ** 2 + fx[None, :, None] ** 2 + fz[None, None, :] ** 2)
    H = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * s2).astype(np.float32)
    return np.fft.irfftn(np.fft.rfftn(cube.astype(np.float32)) * H,
                         s=cube.shape).astype(np.float32)


def _density_com(cube: np.ndarray, origin: np.ndarray, spacing: float) -> np.ndarray:
    """Centre of mass (world A) of a non-negative density cube. Used to centre
    the direct-density rotation expansion on the actual density rather than the
    (possibly off) event centroid."""
    w = np.clip(cube, 0.0, None)
    tot = float(w.sum())
    n = cube.shape[0]
    ax = np.arange(n) * spacing
    if tot <= 0.0:
        return np.asarray(origin, dtype=np.float64) + (n / 2.0) * spacing
    cx = float((w.sum(axis=(1, 2)) * (origin[0] + ax)).sum() / tot)
    cy = float((w.sum(axis=(0, 2)) * (origin[1] + ax)).sum() / tot)
    cz = float((w.sum(axis=(0, 1)) * (origin[2] + ax)).sum() / tot)
    return np.array([cx, cy, cz], dtype=np.float64)


def prepare_target(cube: np.ndarray, origin: np.ndarray, spacing: float,
                   centroid, ligand_radius: float,
                   protein_occupancy: np.ndarray | None = None,
                   taper: float = 2.0, lowpass_sigma: float | None = None):
    """Soft-mask (+ optional protein-zero) the event-map cube, returning the
    POSITIVE masked target and its support mask.

    - soft spherical cosine taper about the centroid at ``ligand_radius`` (+taper
      width); zeros everything outside, no hard edge (a hard edge injects
      high-frequency ripple into the Patterson).
    - HOLE 3b: if ``protein_occupancy`` is given, additionally zero voxels inside
      protein (mirror autobuild.inbuilt.mask_dmap's 1.5 A set_points_around).

    Mean-subtraction is NOT applied here: it is a Patterson-input requirement
    (kill the DC pedestal in the rotation function), whereas the translation /
    Tanimoto step correlates positive densities. fit_conformer_crowther mean-subtracts
    a copy via patterson_input() for the rotation path only.

    If ``lowpass_sigma`` is set, the cube is Gaussian-low-passed first to suppress
    noise-dominated high frequencies (matched-filter / Wiener-lite; helps the
    noise-squaring Patterson and matches the probe bandwidth).
    """
    if lowpass_sigma:
        cube = gaussian_lowpass(cube, spacing, lowpass_sigma)
    n = cube.shape[0]
    ax = (np.arange(n, dtype=np.float64) * spacing)
    gx = origin[0] + ax
    gy = origin[1] + ax
    gz = origin[2] + ax
    cx = np.asarray(centroid, dtype=np.float64)
    R2 = ((gx[:, None, None] - cx[0]) ** 2 +
          (gy[None, :, None] - cx[1]) ** 2 +
          (gz[None, None, :] - cx[2]) ** 2)
    rr = np.sqrt(R2)
    # cosine taper from ligand_radius to ligand_radius + taper
    w = np.clip((ligand_radius + taper - rr) / taper, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * w)  # smootherstep-ish C1 taper
    masked = cube * w.astype(np.float32)

    if protein_occupancy is not None:
        masked = masked * (protein_occupancy <= 0).astype(np.float32)

    support = w > 1e-3
    masked[~support] = 0.0
    return masked.astype(np.float32), support


def patterson_input(masked: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Mean-subtract the masked target within its support so its Patterson has
    no DC pedestal swamping the orientational vectors. Rotation path only."""
    out = masked.copy()
    if support.any():
        out[support] -= out[support].mean()
    out[~support] = 0.0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# The fit -- split into per-EVENT target prep and per-CONFORMER search
# ---------------------------------------------------------------------------
#
# The source (event/z) map is a per-event object; nothing about it depends on
# the conformer. So everything derived from it -- the orthonormal cut, the
# Patterson, the SH expansion, F_target_conj -- is computed ONCE per event in
# prepare_event_target() and reused across all that event's conformers via
# fit_conformer_against(). The conformer's own density is the only thing built
# per conformer, and it is built locally in the cube. This removes the
# per-conformer x parallelism regeneration that the full-cell unmask path (and
# the previous monolithic fit_conformer_crowther) suffered from.


@dataclass
class CrowtherEventTarget:
    """Per-event FRF target: everything derived from the event/z map, computed
    once and reused across the event's conformers."""
    centre: np.ndarray           # event centroid, native Cartesian (A)
    origin: np.ndarray           # cube corner, native Cartesian (A)
    pat_origin: np.ndarray       # Patterson-grid origin (cube-centred)
    f_target: np.ndarray         # SH expansion of the target Patterson
    F_target_conj: np.ndarray    # rfftn(masked positive target).conj()
    target_self: float           # sum(target^2), Tanimoto denominator term
    F_protein_conj: np.ndarray   # rfftn(protein occupancy).conj() (or zeros)


def prepare_event_target(
        target_grid: gemmi.FloatGrid,
        centroid,
        pre: CrowtherPrecompute,
        ligand_radius: float,
        protein_occupancy_grid: gemmi.FloatGrid | None = None,
        lowpass_sigma: float | None = None,
) -> CrowtherEventTarget:
    """Cut + preprocess + Patterson + SH-expand the event/z map ONCE per event.

    # HOLE 1 (target map): pass the event map or the z map as ``target_grid``.
    #   Event map = background-subtracted ligand density (cleaner); z map is the
    #   fallback when bdc is unreliable. Decide + document.
    """
    cfg = pre.config
    n, spacing = cfg.grid, cfg.spacing
    centre = np.asarray(centroid, dtype=np.float64)

    tcube, origin, _ = cut_cube(target_grid, centre, n, spacing)
    prot_occ = None
    if protein_occupancy_grid is not None:
        prot_occ, _, _ = cut_cube(protein_occupancy_grid, centre, n, spacing)
    # tprep: positive masked target (translation/Tanimoto). pat_in: mean-
    # subtracted copy for the Patterson/rotation path only.
    tprep, support = prepare_target(
        tcube, origin, spacing, centre, ligand_radius, prot_occ,
        lowpass_sigma=lowpass_sigma)
    pat_in = patterson_input(tprep, support)
    pat_origin = (-n / 2.0 * spacing * np.ones(3)).astype(np.float32)
    if pre.config.use_patterson:
        # Patterson rotation function (translation-invariant; for poor centroids)
        t_src = rot.compute_patterson(pat_in)
        t_spheres = rot.sample_density_on_spheres(
            t_src, pat_origin, spacing, np.zeros(3, np.float32),
            pre.r, pre.theta, pre.phi)
    else:
        # Direct density overlap (default): expand the mean-subtracted masked
        # density on shells about the masked-density CENTRE OF MASS, not the raw
        # event centroid. The COM is the true density centre regardless of
        # centroid error, so the rotation is computed correctly even when the
        # event centroid is off; the translation FFT then places the pose.
        com = _density_com(tprep, origin, spacing)
        t_spheres = rot.sample_density_on_spheres(
            pat_in, origin.astype(np.float32), spacing,
            com.astype(np.float32), pre.r, pre.theta, pre.phi)
    f_target = rot.sh_expand_fast(t_spheres, pre.Y_conj)

    # raw (un-Pattersoned) positive target for the translation FFT + Tanimoto
    F_target_conj = np.fft.rfftn(tprep).conj()
    target_self = float((tprep * tprep).sum())
    if prot_occ is not None:
        F_protein_conj = np.fft.rfftn(prot_occ.astype(np.float32)).conj()
    else:
        F_protein_conj = np.zeros_like(F_target_conj)

    del tcube, prot_occ, tprep, pat_in, t_spheres, support
    return CrowtherEventTarget(
        centre=centre, origin=origin.astype(np.float64), pat_origin=pat_origin,
        f_target=f_target, F_target_conj=F_target_conj,
        target_self=target_self, F_protein_conj=F_protein_conj)


def fit_conformer_against(
        target: CrowtherEventTarget,
        conformer: gemmi.Structure,
        pre: CrowtherPrecompute,
        sigma: float | None = None,
        n_candidates: int = 1,
):
    """Per-conformer FRF search against a prepared CrowtherEventTarget. Only the
    conformer's own density is built here, locally in the cube.

    With ``n_candidates == 1`` (default) returns the single Tanimoto-best
    ``(optimized_structure, tanimoto, pose_centroid)``. With ``n_candidates > 1``
    returns a list of the top-N such tuples (by Tanimoto) so the caller can
    re-rank them with a different objective (e.g. CNN-arbitrate, matching the DE
    path's best-of-N-by-CNN behaviour).
    """
    cfg = pre.config
    n, spacing = cfg.grid, cfg.spacing
    # sigma decoupled from the (sigma-independent) precompute so a cached pre is
    # reusable across datasets at different resolutions.
    sig = cfg.sigma if sigma is None else sigma
    centre, origin, pat_origin = target.centre, target.origin, target.pat_origin
    pat_centre = np.zeros(3, dtype=np.float32)

    # probe: heavy-atom coords + Z weights, centred at origin
    coords, weights = _heavy_atoms(conformer)
    coords = coords - coords.mean(axis=0, keepdims=True)
    stamp, r_vox = vox.make_gaussian_stamp(sig, spacing)

    # probe density -> shells -> SH; must match the target's rotation mode.
    probe0 = vox.voxelise_gaussian(coords + centre[None, :], origin, spacing,
                                   n, stamp, r_vox, weights=weights)
    if cfg.use_patterson:
        p_src = rot.compute_patterson(probe0)
        p_spheres = rot.sample_density_on_spheres(
            p_src, pat_origin, spacing, pat_centre, pre.r, pre.theta, pre.phi)
    else:
        p_spheres = rot.sample_density_on_spheres(
            probe0, origin.astype(np.float32), spacing,
            centre.astype(np.float32), pre.r, pre.theta, pre.phi)
    f_probe = rot.sh_expand_fast(p_spheres, pre.Y_conj)
    X_l = rot.cross_corr_tensor(target.f_target, f_probe, pre.r, pre.dr, cfg.L_max)
    scores = rot.score_all_rotations(X_l, pre.D_batch)
    del probe0, p_spheres, f_probe, X_l  # consumed

    # top-K orientations -> translation FFT + clash Tanimoto; collect candidates
    top_k = min(cfg.top_k, cfg.n_rotations)
    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    cands = []
    for idx in top_idx:
        R = pre.Rs_mat[idx]
        rot_coords = coords @ R.T + centre[None, :]
        pg = vox.voxelise_gaussian(rot_coords, origin, spacing, n, stamp, r_vox,
                                   weights=weights)
        F_p = np.fft.rfftn(pg)
        probe_self = float((pg * pg).sum())
        pose = refine_translation_with_clash(
            F_p, target.F_target_conj, target.F_protein_conj,
            target.target_self, probe_self, cfg.lambda_clash, n)
        cands.append((pose.combined, R, pose))

    cands.sort(key=lambda c: c[0], reverse=True)

    def _place(R, pose):
        shift = _voxel_to_shift(pose.best_translation_voxel, n, spacing)
        fc = coords @ R.T + centre[None, :] + shift[None, :]
        return (_place_structure(conformer, fc),
                float(pose.tanimoto_at_best_combined),
                tuple(fc.mean(axis=0)))

    if n_candidates <= 1:
        _c, R, pose = cands[0]
        return _place(R, pose)
    return [_place(R, pose) for _c, R, pose in cands[:n_candidates]]


def fit_conformer_crowther(
        centroid,
        conformer: gemmi.Structure,
        target_grid: gemmi.FloatGrid,
        pre: CrowtherPrecompute,
        ligand_radius: float,
        protein_occupancy_grid: gemmi.FloatGrid | None = None,
        sigma: float | None = None,
):
    """Convenience single-conformer entry: prepare the event target then fit one
    conformer. For multiple conformers of the same event, call
    prepare_event_target() once and fit_conformer_against() per conformer to
    avoid re-deriving the per-event target."""
    target = prepare_event_target(
        target_grid, centroid, pre, ligand_radius, protein_occupancy_grid)
    return fit_conformer_against(target, conformer, pre, sigma=sigma)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Minimal element->Z table for probe weighting; fragments are HCNOSP + halogens.
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
      "CL": 17, "BR": 35, "I": 53}


def _heavy_atoms(structure: gemmi.Structure):
    """Heavy-atom Cartesian coords (N,3) and Z weights (N,), in iteration order.

    # HOLE 2 (atom order): write-back assumes _place_structure visits atoms in
    # this same order and that H were excluded from both. get_conformers builds
    # H-free LIG residues, so this holds for the current conformer source -- but
    # assert len(coords) == n_heavy in _place_structure.
    """
    coords, weights = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    p = atom.pos
                    coords.append([p.x, p.y, p.z])
                    weights.append(_Z.get(atom.element.name.upper(), 6))
    return (np.asarray(coords, dtype=np.float64),
            np.asarray(weights, dtype=np.float32))


def _voxel_to_shift(voxel, n: int, spacing: float) -> np.ndarray:
    """Map a cyclic translation-grid index to the Cartesian shift to ADD to the
    probe coords so it aligns with the target.

    refine_translation_with_clash maximises cc[s] = irfftn(F_probe.conj(F_target)),
    whose peak satisfies target[i-s] ~ probe[i]; i.e. target == probe shifted by
    +s. To move the probe onto the target we therefore add -s. The cyclic index
    is unwrapped to a signed lag first (a peak past n/2 is a negative lag).

    Verified against a plant-and-recover round trip in
    tests/test_crowther_fit.py::test_translation_convention.
    """
    v = np.asarray(voxel, dtype=np.int64)
    signed = ((v + n // 2) % n) - n // 2
    return (-signed * spacing).astype(np.float64)


def _place_structure(conformer: gemmi.Structure, coords: np.ndarray) -> gemmi.Structure:
    """Clone ``conformer`` and set its heavy-atom positions to ``coords`` (native
    Cartesian A -- no inverse-cell transform needed, the cube was in native
    frame). Visits atoms in the same order as _heavy_atoms."""
    st = conformer.clone()
    k = 0
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    x, y, z = coords[k]
                    atom.pos = gemmi.Position(float(x), float(y), float(z))
                    k += 1
    if k != coords.shape[0]:
        raise ValueError(f"atom-order mismatch: placed {k} of {coords.shape[0]}")
    return st
