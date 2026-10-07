"""SH-Crowther fast-rotation-function ligand fitting for PanDDA2 event maps.

An experimental, deterministic replacement for the
``scipy.optimize.differential_evolution`` pose search in
``pandda_gemmi.autobuild.inbuilt.score_conformer``, enabled with
``PANDDA_CROWTHER_FIT=1``. The conformer's density is expanded in spherical
harmonics about the event and rotated against the z map with a Crowther (1972)
fast rotation function; the top rotations are positioned by an FFT translation
function, locally refined on the DE score grid, and ranked by the same build
CNN the DE path uses, so the return contract downstream is unchanged.

Layout
------
- ``rotation``    : the gemmi-free spherical-harmonic core. Assumes an
                    orthonormal P1 cube (isotropic scalar spacing, Cartesian
                    origin, C-order grid).
- ``voxelise``    : Gaussian stamping of a conformer onto the cube, with
                    optional per-atom Z weighting.
- ``translation`` : FFT translation function + clash-penalised Tanimoto.
- ``fit``         : the PanDDA2 adapter. Cuts the orthonormal cube from the
                    event/z map, preprocesses the target, drives the search,
                    and writes the pose back into the dataset's native frame.

Design notes
------------
- The rotation function correlates the density directly (about the masked
  density's centre of mass), not its Patterson: on weak difference density the
  Patterson's squaring amplified noise and diverged from an exhaustive search,
  whereas direct overlap reproduces the exhaustive search exactly.
- Defaults are deliberately lean (band limit L=8, ~2000 rotations): on real
  data the residual placement error is scoring-limited, not sampling-limited,
  so denser sampling buys nothing and costs ~100x the precompute memory.
- The FRF is a seed (~1 A rigid placement); the per-seed local refinement on
  the score grid is what gets to sub-A.

Lifted from the author's FragVol orthonormal-MR engine.
"""
