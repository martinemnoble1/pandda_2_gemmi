"""Benchmark: SH-Crowther FRF vs brute-force rotation search.

Apples-to-apples on the synthetic arbitrary-rotation scenarios from
test_sht_fit: both methods use the SAME SO(3) sample set and the SAME
translation/Tanimoto scoring (refine_translation_with_clash). The only
difference is the rotation search --

  brute : evaluate a translation FFT for EVERY rotation, keep the best
          (the accuracy ceiling over this rotation set, FragVol inspect_mr_brute).
  FRF   : rank all rotations by the O(L^3) SH rotation function, then run the
          translation FFT only on the top-K.

Reports per-conformer accuracy (RMSD-to-truth) and timing, and the FRF one-off
precompute cost (amortised across all conformers a worker handles). Run with::

    pytest tests/test_sht_benchmark.py -s -q

The asserts are deliberately loose (this is a benchmark, not a unit test): FRF
must not be meaningfully less accurate than exhaustive brute, and must be
faster per conformer.
"""

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pandda_gemmi.autobuild.sht import rotation as rot, voxelise as vox
from pandda_gemmi.autobuild.sht.translation import refine_translation_with_clash
from pandda_gemmi.autobuild.sht.fit import (
    ShtConfig, build_precompute, cut_cube, prepare_target, patterson_input,
    fit_conformer_sht, _voxel_to_shift, _heavy_atoms,
)
from tests.test_sht_fit import _COORDS, _gemmi_structure, _density_grid, _heavy_coords


_START_ROTATIONS = [
    (0.0, 0.0, 0.0),
    (55.0, -30.0, 80.0),
    (120.0, 70.0, -40.0),
    (200.0, 15.0, -95.0),
]
_T_TRUE = np.array([1.5, -1.0, 0.5])


def _brute_fit(centre, conformer, target_grid, pre, ligand_radius):
    """Exhaustive rotation search over pre.Rs_mat with per-rotation translation
    FFT + Tanimoto. Same scoring + write-back as fit_conformer_sht; differs only
    in evaluating every rotation instead of an SH-ranked top-K."""
    cfg = pre.config
    n, spacing = cfg.grid, cfg.spacing
    centre = np.asarray(centre, dtype=np.float64)

    tcube, origin, _ = cut_cube(target_grid, centre, n, spacing)
    tprep, support = prepare_target(tcube, origin, spacing, centre, ligand_radius)
    F_target_conj = np.fft.rfftn(tprep).conj()
    target_self = float((tprep * tprep).sum())
    F_protein_conj = np.zeros_like(F_target_conj)

    coords, weights = _heavy_atoms(conformer)
    coords = coords - coords.mean(axis=0, keepdims=True)
    stamp, r_vox = vox.make_gaussian_stamp(cfg.sigma, spacing)

    best_combined, best = -np.inf, None
    for idx in range(cfg.n_rotations):
        R = pre.Rs_mat[idx]
        pg = vox.voxelise_gaussian(coords @ R.T + centre[None, :], origin,
                                   spacing, n, stamp, r_vox, weights=weights)
        F_p = np.fft.rfftn(pg)
        probe_self = float((pg * pg).sum())
        pose = refine_translation_with_clash(
            F_p, F_target_conj, F_protein_conj, target_self, probe_self, 0.0, n)
        if pose.combined > best_combined:
            best_combined, best = pose.combined, (R, pose)

    R, pose = best
    shift = _voxel_to_shift(pose.best_translation_voxel, n, spacing)
    final = coords @ R.T + centre[None, :] + shift[None, :]
    return final, float(pose.tanimoto_at_best_combined)


@pytest.mark.parametrize("grid,n_rot", [(32, 2000)])
def test_benchmark_frf_vs_brute(grid, n_rot, capsys):
    cfg = ShtConfig(grid=grid, spacing=0.5, L_max=12, n_r=14,
                    n_rotations=n_rot, sigma=1.0, top_k=30)

    t0 = time.perf_counter()
    pre = build_precompute(cfg)
    precompute_s = time.perf_counter() - t0

    centre = np.array([20.0, 20.0, 20.0])
    truth = _COORDS + centre + _T_TRUE
    target_grid = _density_grid(truth)

    rows = []
    for euler in _START_ROTATIONS:
        R_start = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        conformer = _gemmi_structure(_COORDS @ R_start.T)

        t0 = time.perf_counter()
        st, frf_score, _c = fit_conformer_sht(
            centre, conformer, target_grid, pre, ligand_radius=6.0)
        frf_s = time.perf_counter() - t0
        frf_rmsd = np.sqrt(((_heavy_coords(st) - truth) ** 2).sum(1).mean())

        t0 = time.perf_counter()
        bcoords, brute_score = _brute_fit(
            centre, conformer, target_grid, pre, ligand_radius=6.0)
        brute_s = time.perf_counter() - t0
        brute_rmsd = np.sqrt(((bcoords - truth) ** 2).sum(1).mean())

        rows.append((euler, frf_rmsd, brute_rmsd, frf_s, brute_s))

    # report
    lines = [
        "",
        f"  FRF precompute (one-off / worker): {precompute_s:.2f} s  "
        f"(D_batch {sum(a.nbytes for a in pre.D_batch)/1e6:.0f} MB)",
        f"  grid={grid}^3  n_rotations={n_rot}  L={cfg.L_max}  top_k={cfg.top_k}",
        "",
        f"  {'start euler':>18}  {'FRF rmsd':>8}  {'brute rmsd':>10}  "
        f"{'FRF s':>7}  {'brute s':>8}  {'speedup':>7}",
    ]
    sp = []
    for euler, fr, br, fs, bs in rows:
        sp.append(bs / fs)
        lines.append(
            f"  {str(euler):>18}  {fr:>8.2f}  {br:>10.2f}  "
            f"{fs:>7.3f}  {bs:>8.3f}  {bs/fs:>6.1f}x")
    lines.append("")
    lines.append(f"  mean per-conformer speedup: {np.mean(sp):.1f}x")
    report = "\n".join(lines)
    with capsys.disabled():
        print(report)

    frf_rmsds = np.array([r[1] for r in rows])
    brute_rmsds = np.array([r[2] for r in rows])
    frf_times = np.array([r[3] for r in rows])
    brute_times = np.array([r[4] for r in rows])

    # accuracy: FRF (top-K) must not be meaningfully worse than exhaustive brute
    assert np.all(frf_rmsds < brute_rmsds + 0.75), \
        f"FRF rmsd {frf_rmsds} vs brute {brute_rmsds}"
    assert np.all(frf_rmsds < 1.5), f"FRF rmsds {frf_rmsds}"
    # timing: FRF faster per conformer
    assert frf_times.mean() < brute_times.mean()
