"""PANDDA_LOCAL_AUTOBUILD must reproduce the full-cell autobuild_conformer.

End-to-end on a synthetic monoclinic dataset: the same event, conformer and
seed are built once through the stock path (full-cell ``unmask``) and once
through the local-box path, and the two builds are compared. This is the
behaviour-preservation claim for the memory change: identical result contract,
same pose (to within the DE's own noise), and -- the point of the exercise --
the local path never densifies the cell.

The CNN scorer is replaced by a translation-invariant density-overlap stub so
the test needs no checkpoint and runs anywhere.
"""

import functools

import numpy as np
import gemmi
import pytest

import pandda_gemmi.autobuild.inbuilt as ib


# --- synthetic dataset -----------------------------------------------------

# Asymmetric heavy-atom cluster (no accidental symmetry to mask a wrong pose).
_LIG = np.array([
    [0.0, 0.0, 0.0],
    [2.3, 0.0, 0.0],
    [0.0, 3.1, 0.0],
    [0.0, 0.0, 4.2],
    [1.4, 1.1, 0.3],
    [-1.8, 0.6, 2.0],
    [3.5, 1.4, 0.9],
    [1.2, 4.3, 1.5],
    [-1.0, 2.2, 3.8],
    [2.6, 2.7, 3.1],
    [-2.9, -0.9, 0.4],
    [0.8, -1.6, 2.6],
], dtype=np.float64)

CELL = (56.0, 58.0, 60.0, 90.0, 95.0, 90.0)
SHAPE = (112, 116, 120)          # ~0.5 A native spacing


def _gemmi_structure(coords, cell):
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = "P 1"
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
    st.setup_entities()
    return st


def _protein_stub(cell, near):
    """One ALA CA 4 A from the ligand so get_contacts has something to count."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId(10, " ")
    a = gemmi.Atom()
    a.name = "CA"
    a.element = gemmi.Element("C")
    a.pos = gemmi.Position(near[0] + 4.0, near[1], near[2])
    res.add_atom(a)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


class _Mask:
    def __init__(self, indicies):
        self.indicies = indicies


class _Frame:
    """Minimal DFrame: every native voxel is in the mask, and ``unmask`` counts
    how often the full cell is densified."""

    def __init__(self, cell, shape):
        self.unit_cell = cell
        self.spacing = shape
        self.spacegroup = 1
        self.mask = _Mask(np.nonzero(np.ones(shape, dtype=np.int8)))
        self.unmask_calls = 0

    def get_grid(self):
        g = gemmi.FloatGrid(*self.spacing)
        g.set_unit_cell(gemmi.UnitCell(*self.unit_cell))
        g.spacegroup = gemmi.SpaceGroup("P 1")
        return g

    def unmask(self, sparse):
        self.unmask_calls += 1
        g = self.get_grid()
        np.array(g, copy=False)[self.mask.indicies] = np.asarray(
            getattr(sparse, "data", sparse), dtype=np.float32)
        return g

    def mask_grid(self, grid):
        from pandda_gemmi.dmaps import SparseDMap
        return SparseDMap(np.array(grid, copy=False)[self.mask.indicies])


def _gaussian_density(frame, coords, sigma=1.2):
    """Dense native array with unit Gaussians at Cartesian ``coords``."""
    cell = gemmi.UnitCell(*frame.unit_cell)
    nu, nv, nw = frame.spacing
    cols = [cell.orthogonalize(gemmi.Fractional(1, 0, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 1, 0)),
            cell.orthogonalize(gemmi.Fractional(0, 0, 1))]
    M = np.array([[c.x, c.y, c.z] for c in cols]).T
    fu, fv, fw = np.meshgrid(np.arange(nu) / nu, np.arange(nv) / nv,
                             np.arange(nw) / nw, indexing="ij")
    cart = np.stack([fu, fv, fw], axis=-1) @ M.T
    arr = np.zeros((nu, nv, nw), dtype=np.float32)
    for atom in coords:
        d2 = ((cart - atom) ** 2).sum(axis=-1)
        arr += np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
    return arr


def _score_build_stub(structure, z_grid, xmap_grid):
    """Translation-invariant CNN stand-in: mean z density at the heavy atoms."""
    vals = [z_grid.interpolate_value(atom.pos)
            for model in structure for chain in model
            for res in chain for atom in res if atom.element.name != "H"]
    return float(np.mean(vals)), None


def _heavy_coords(st):
    return np.array([[a.pos.x, a.pos.y, a.pos.z]
                     for m in st for c in m for r in c for a in r
                     if a.element.name != "H"])


def _rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


class _Conformer:
    def __init__(self, structure):
        self.structure = structure


class _Dataset:
    def __init__(self, structure):
        self.structure = structure


@pytest.fixture(scope="module")
def synthetic():
    frame = _Frame(CELL, SHAPE)
    cell = gemmi.UnitCell(*CELL)
    centre = np.array(cell.orthogonalize(gemmi.Fractional(0.5, 0.5, 0.5)).tolist())
    true_coords = _LIG - _LIG.mean(0) + centre
    ligand = _gaussian_density(frame, true_coords)
    # Ground-state feature overlapping the site, so the BDC has a genuine
    # optimum: dtag = (1 - bdc_true) * ligand + bdc_true * mean, bdc_true = 0.5.
    ground = 1.5 * _gaussian_density(frame, [centre + np.array([2.0, -1.5, 1.0])],
                                     sigma=2.5)
    rng = np.random.default_rng(0)
    noise = 0.02 * rng.standard_normal(ligand.shape).astype(np.float32)
    dense = (0.5 * ligand + 0.5 * ground + noise).astype(np.float32)
    sparse = dense[frame.mask.indicies]
    mean = (ground + noise)[frame.mask.indicies].astype(np.float32)
    z = ligand[frame.mask.indicies]

    # Start from an arbitrary orientation; score_conformer recentres on the
    # (slightly off) event centroid so only the orientation matters.
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [70, -40, 115], degrees=True).as_matrix()
    start = (_LIG - _LIG.mean(0)) @ R.T + centre
    return dict(
        frame=frame,
        centroid=centre + np.array([0.4, -0.3, 0.5]),
        true_coords=true_coords,
        conformer=_Conformer(_gemmi_structure(start, CELL)),
        protein=_Dataset(_protein_stub(CELL, centre)),
        xmap=sparse.astype(np.float32),
        mean=mean,
        z=(5.0 * z).astype(np.float32),
        dense=dense,
    )


def _build(synthetic, tmp_path, tag):
    s = synthetic
    out = tmp_path / tag
    out.mkdir()
    result = ib.autobuild_conformer(
        s["centroid"], 0.5, s["conformer"],
        s["xmap"], s["mean"], s["frame"], out, 0, 2.0, s["protein"],
        s["xmap"], s["mean"], s["z"], s["xmap"], _score_build_stub, s["dense"],
    )
    (path, rec), = result.items()
    built = gemmi.read_structure(path)
    return rec, built


def test_local_matches_full_cell(synthetic, tmp_path, monkeypatch):
    # Same seed for both runs so the DE randomness is not a confound; fewer DE
    # restarts so the test is quick (both paths get the same reduction).
    monkeypatch.setenv("PANDDA_DE_SEED", "7")
    monkeypatch.setattr(ib, "score_conformer",
                        functools.partial(ib.score_conformer, event_fit_num_trys=2))
    frame = synthetic["frame"]

    monkeypatch.delenv("PANDDA_LOCAL_AUTOBUILD", raising=False)
    frame.unmask_calls = 0
    full_rec, full_built = _build(synthetic, tmp_path, "full")
    assert frame.unmask_calls > 0

    monkeypatch.setenv("PANDDA_LOCAL_AUTOBUILD", "1")
    frame.unmask_calls = 0
    local_rec, local_built = _build(synthetic, tmp_path, "local")
    assert frame.unmask_calls == 0, "local path densified the full cell"

    # Same result contract.
    assert set(local_rec) == set(full_rec)
    for k in ("score", "local_signal", "new_bdc", "noise", "signal",
              "num_points", "optimal_contour", "num_contacts"):
        assert np.isfinite(local_rec[k]), k

    # Same build. The two DE runs see the same objective up to the local box's
    # resampling, so poses agree to well under the ~1 A the DE itself scatters
    # over restarts; tolerances leave room for platform float differences.
    true_centroid = synthetic["true_coords"].mean(0)
    assert _rmsd(_heavy_coords(local_built), _heavy_coords(full_built)) < 2.0
    assert np.linalg.norm(np.asarray(local_rec["centroid"])
                          - np.asarray(full_rec["centroid"])) < 1.5
    for rec in (full_rec, local_rec):      # both in the event, not wandered off
        assert np.linalg.norm(np.asarray(rec["centroid"]) - true_centroid) < 6.0

    # Same scores: the BDC, the corrected-map signal/noise sums and the contour
    # are all computed on the local box in the local path and must match the
    # full-cell values, not merely be finite.
    assert local_rec["new_bdc"] == pytest.approx(full_rec["new_bdc"], abs=0.05)
    assert local_rec["score"] == pytest.approx(full_rec["score"], rel=0.15)
    for k in ("noise", "signal", "optimal_contour"):
        assert local_rec[k] == pytest.approx(full_rec[k], rel=0.15), k
    assert abs(local_rec["num_points"] - full_rec["num_points"]) <= 5
    assert local_rec["num_contacts"] == full_rec["num_contacts"]

    # The local build is written in the dataset cell, not the box cell.
    assert local_built.cell.a == pytest.approx(CELL[0])
    assert local_built.cell.beta == pytest.approx(CELL[4])


def test_local_flag_is_parsed_not_presence_checked(synthetic, tmp_path, monkeypatch):
    """PANDDA_LOCAL_AUTOBUILD=0 must mean OFF (stock full-cell path)."""
    monkeypatch.setenv("PANDDA_DE_SEED", "7")
    monkeypatch.setattr(ib, "score_conformer",
                        functools.partial(ib.score_conformer, event_fit_num_trys=1))
    monkeypatch.setenv("PANDDA_LOCAL_AUTOBUILD", "0")
    frame = synthetic["frame"]
    frame.unmask_calls = 0
    _build(synthetic, tmp_path, "off")
    assert frame.unmask_calls > 0
