"""Manual real-data check: run the SH-Crowther FRF against a real event z-map
from a completed PanDDA run, and report pose / timing / memory.

Generic and dataset-agnostic -- point it at any PanDDA processed_datasets dir.
Use a PUBLIC dataset (e.g. the BAZ2B PanDDA demo data from Zenodo) for anything
shareable. Not a pytest; a driver. Run::

    python tests/crowther_realdata_check.py <processed_datasets_dir> <dtag> [event_id]
    # or set PANDDA_PROCESSED_DIR + PANDDA_DTAG

Expects the standard per-dataset layout:
    <dir>/<dtag>/<dtag>-z_map.native.ccp4
    <dir>/<dtag>/events.yaml
    <dir>/<dtag>/ligand_files/<*.cif>
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import gemmi
import yaml

from pandda_gemmi.fs.pandda_input import LigandFiles
from pandda_gemmi.autobuild.inbuilt import get_conformers
from pandda_gemmi.autobuild.crowther.fit import (
    CrowtherConfig, get_precompute, prepare_event_target, fit_conformer_against,
    sigma_from_resolution,
)


def event_centroid(events_yaml, event_id):
    with open(events_yaml) as f:
        events = yaml.safe_load(f)
    pos = np.array(events[event_id]["Position Array"], dtype=np.float64)
    return pos.mean(axis=0)


def main(processed_dir, dtag, event_id=1, resolution=2.5):
    ddir = Path(processed_dir) / dtag
    z_grid = gemmi.read_ccp4_map(str(ddir / f"{dtag}-z_map.native.ccp4")).grid
    cell = z_grid.unit_cell
    centroid = event_centroid(ddir / "events.yaml", event_id)
    dense_mb = z_grid.nu * z_grid.nv * z_grid.nw * 4 / 1e6
    print(f"{dtag} event {event_id}")
    print(f"  z-map {z_grid.nu}x{z_grid.nv}x{z_grid.nw}  cell a={cell.a:.1f} "
          f"b={cell.b:.1f} c={cell.c:.1f} beta={cell.beta:.1f}  ({dense_mb:.0f} MB dense)")
    print(f"  event centroid (native A): {centroid.round(2)}")

    cif = next((ddir / "ligand_files").glob("*.cif"), None)
    conformers = get_conformers(LigandFiles(cif, None, None)) if cif else {}
    if not conformers:
        print("  NO CONFORMERS (check the ligand cif) -- abort")
        return 1
    conf = conformers[0]
    coords = np.array([[a.pos.x, a.pos.y, a.pos.z]
                       for m in conf for ch in m for r in ch for a in r
                       if a.element.name != "H"])
    radius = float(np.linalg.norm(coords - coords.mean(0), axis=1).max())
    print(f"  ligand: {coords.shape[0]} heavy atoms, radius {radius:.1f} A")
    if radius < 1.0:
        print("  WARNING: degenerate conformer (radius ~0) -- embedding likely "
              "failed for this ligand; FRF result will be meaningless")

    cfg = CrowtherConfig()
    sigma = sigma_from_resolution(resolution)
    print(f"  config grid={cfg.grid} L={cfg.L_max} N={cfg.n_rotations} "
          f"cube={cfg.grid*cfg.spacing:.0f}A sigma={sigma:.2f}")

    t0 = time.perf_counter()
    pre = get_precompute(cfg)
    print(f"  precompute (one-off/worker): {time.perf_counter()-t0:.1f} s, "
          f"D_batch {sum(a.nbytes for a in pre.D_batch)/1e6:.0f} MB")

    tracemalloc.start()
    t0 = time.perf_counter()
    target = prepare_event_target(z_grid, centroid, pre, ligand_radius=radius + 2.0)
    t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    st, tani, pose_c = fit_conformer_against(target, conf, pre, sigma=sigma)
    t_fit = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  prepare_event_target {t_prep:.3f}s  fit_conformer_against {t_fit:.3f}s")
    print(f"  transient peak {peak/1e6:.1f} MB (vs {dense_mb:.0f} MB for one dense unmask)")
    print(f"  Tanimoto {tani:.3f}  pose centroid {np.array(pose_c).round(2)}  "
          f"|pose-event| {np.linalg.norm(np.array(pose_c)-centroid):.2f} A")

    out = ddir / f"{dtag}_event{event_id}_crowther_fit.pdb"
    st.setup_entities()
    st.write_pdb(str(out))
    print(f"  wrote {out} -- inspect against the z-map in coot")
    return 0


if __name__ == "__main__":
    pdir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PANDDA_PROCESSED_DIR")
    dtag = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PANDDA_DTAG")
    eid = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if not pdir or not dtag:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(pdir, dtag, eid))
