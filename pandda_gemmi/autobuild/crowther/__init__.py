"""SH-Crowther fast-rotation-function ligand fitting for PanDDA2 event maps.

This package lifts the FragVol orthonormal-MR engine (a spherical-harmonic
Crowther 1972 fast rotation function + FFT translation function + Tanimoto
scoring) into PanDDA2's autobuild path. It is a faster, deterministic
replacement for the `scipy.optimize.differential_evolution` pose search in
``pandda_gemmi.autobuild.inbuilt.score_conformer``.

Layout
------
- ``rotation``    : the gemmi-free SH core, lifted verbatim from FragVol
                    ``inspect_mr_sh.py``. Assumes the orthonormal-P1 invariant
                    (isotropic scalar spacing, Cartesian origin, C-order grid).
- ``voxelise``    : Gaussian "calc_fc-lite" stamping of a conformer onto the
                    cube, with optional per-atom Z weighting.
- ``translation`` : FFT translation function + clash-penalised Tanimoto,
                    lifted from FragVol ``inspect_clash.py``.
- ``fit``         : the PanDDA2 adapter. Cuts the orthonormal cube from the
                    event/z map via ``SampleFrame``, preprocesses the target,
                    drives the search, and writes the pose back into the
                    dataset's native Cartesian frame. This is where the
                    decisions still to be made are marked ``# HOLE``.

Provenance: FragVol @ ~/Developer/FragVol (inspect_mr_sh.py, inspect_clash.py,
fragvol.py), as of 2026-06-08. The science scoping lives in that repo's
PANDDA2_BRIEFING.md; the corrections agreed during integration (keep the
Patterson for the rotation function; keep the 2.5x clean-box headroom; mask +
mean-subtract the experimental target before the Patterson) are encoded here.
"""
