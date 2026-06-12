import itertools
import os
import time

import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import scipy
from scipy import spatial as spsp, optimize

from pandda_gemmi import constants
from ..interfaces import *

from ..fs import try_make
from ..dmaps import load_dmap, save_dmap, SparseDMap
from ..dataset.structure import save_structure, load_structure, Structure
from ..dataset.small import (
    get_comp_block_key,
    get_fragment_mol_from_dataset_cif_path,
)
from .autobuild import AutobuildResult


def get_fragment_mol_from_dataset_smiles_path(dataset_smiles_path: Path):
    smiles_path = dataset_smiles_path

    # Get smiels string
    with open(str(smiles_path), "r") as f:
        smiles_string: str = str(f.read())

    # Load the mol
    m: Chem.Mol = Chem.MolFromSmiles(smiles_string)

    return m


# NOTE: get_fragment_mol_from_dataset_cif_path now lives in
# pandda_gemmi.dataset.small (imported above) -- a single gemmi-based
# implementation that reads every monomer-dictionary dialect. The previous
# duplicate copy here only understood the legacy `_chem_comp_bond.type`
# column and silently produced bondless (collapsed) ligands from acedrg
# PDBx dicts.


def get_structures_from_mol(mol: Chem.Mol, dataset_cif_path, max_conformers):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Find the relevant atoms loop
    key = get_comp_block_key(cif)
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))

    fragment_structures = {}
    for i, conformer in enumerate(mol.GetConformers()):

        positions: np.ndarray = conformer.GetPositions()

        structure: gemmi.Structure = gemmi.Structure()
        model: gemmi.Model = gemmi.Model(f"{i}")
        chain: gemmi.Chain = gemmi.Chain(f"{i}")
        residue: gemmi.Residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(1, ' ')

        # Loop over atoms, adding them to a gemmi residue
        for j, atom in enumerate(mol.GetAtoms()):
            # Get the atomic symbol
            atom_symbol: str = atom.GetSymbol()
            # print(f"{j} : {atom_symbol}")

            # if atom_symbol == "H":
            #     continue
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            # gemmi_atom.name = atom_symbol
            gemmi_atom.name = atom_id_loop[j]
            gemmi_atom.pos = gemmi_pos
            gemmi_atom.element = gemmi_element

            # Add atom to residue
            residue.add_atom(gemmi_atom)

        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        fragment_structures[i] = structure

        if len(fragment_structures) > max_conformers:
            return fragment_structures

    return fragment_structures


def _de_seed():
    """Optional fixed RNG seed for the stochastic autobuild steps
    (differential_evolution + RDKit conformer embedding), from PANDDA_DE_SEED.
    Returns int when set, else None (scipy/RDKit default = unseeded). Setting it
    makes autobuild deterministic so e.g. local-grid vs full-grid runs can be
    compared without the DE-randomness confound."""
    v = os.environ.get("PANDDA_DE_SEED")
    return int(v) if v not in (None, "") else None


def _max_ligand_heavy_atoms():
    # A fragment-screening ligand is a small molecule (typically < ~50 heavy
    # atoms; a ~15-residue peptide is still < 150). Anything far above this is
    # not a ligand -- almost always a model/protein file mis-detected as one via
    # a permissive ligand regex. Building it as a fragment is meaningless and, on
    # a large complex, exhausts memory (it OOM'd a 128 GB box). Tunable.
    return int(os.environ.get("PANDDA_MAX_LIGAND_ATOMS", 150))


def _structure_heavy_atom_count(st):
    return sum(1 for model in st for chain in model for residue in chain
               for atom in residue if atom.element.name != "H")


def get_conformers(
        ligand_files: LigandFilesInterface,
        pruning_threshold=1.5,
        num_pose_samples=1000,
        max_conformers=10,
):
    # Decide how to load

    if ligand_files.ligand_cif is not None:
        mol = get_fragment_mol_from_dataset_cif_path(ligand_files.ligand_cif)

        # Guard: refuse to treat a non-fragment (e.g. a whole protein/model
        # mis-detected as a ligand) as a buildable ligand -- before the
        # expensive conformer embedding. See _max_ligand_heavy_atoms.
        if mol is None:
            return {}
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > _max_ligand_heavy_atoms():
            print(f"Ligand from {ligand_files.ligand_cif.name} has {n_heavy} heavy "
                  f"atoms (> {_max_ligand_heavy_atoms()} fragment limit): skipping. "
                  f"This is almost certainly a model/protein file mis-detected as a "
                  f"ligand (check your --ligand_cif_regex / --ligand_pdb_regex).")
            return {}

        # Generate conformers
        # mol.CalcImplicitValence()
        # mol: Chem.Mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_pose_samples,
            pruneRmsThresh=pruning_threshold,
            randomSeed=(_de_seed() if _de_seed() is not None else -1),
        )

        # Translate to structures
        fragment_structures = get_structures_from_mol(
            mol,
            ligand_files.ligand_cif,
            max_conformers,
        )

        return fragment_structures

    # if ligand_files.ligand_smiles is not None:
    #     # print(f"\t\t\tGetting conformers from: {ligand_files.ligand_smiles}")
    #     mol = get_fragment_mol_from_dataset_smiles_path(ligand_files.ligand_smiles)
    #
    #     # Generate conformers
    #     mol: Chem.Mol = Chem.AddHs(mol)
    #
    #     # Generate conformers
    #     cids = AllChem.EmbedMultipleConfs(
    #         mol,
    #         numConfs=num_pose_samples,
    #         pruneRmsThresh=pruning_threshold)
    #
    #     # Translate to structures
    #     fragment_structures = get_structures_from_mol(
    #         mol,
    #         max_conformers,
    #     )
    #
    #     return fragment_structures

    if ligand_files.ligand_pdb is not None:
        # print(f"\t\t\tGetting conformers from: {ligand_files.ligand_pdb}")

        st = load_structure(ligand_files.ligand_pdb)
        st = getattr(st, "structure", st)

        # Same guard for the pdb path: a whole-protein "ligand" pdb is not a
        # fragment and must not be built/docked (memory blow-up on large cells).
        n_heavy = _structure_heavy_atom_count(st)
        if n_heavy > _max_ligand_heavy_atoms():
            print(f"Ligand from {ligand_files.ligand_pdb.name} has {n_heavy} heavy "
                  f"atoms (> {_max_ligand_heavy_atoms()} fragment limit): skipping. "
                  f"This is almost certainly a model/protein file mis-detected as a "
                  f"ligand (check your --ligand_pdb_regex).")
            return {}

        fragment_structures = {0: st, }

        return fragment_structures


    else:
        return {}


def get_structure_mean(structure):
    xs = []
    ys = []
    zs = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom.pos)
                    pos: gemmi.Position = atom.pos
                    xs.append(pos.x)
                    ys.append(pos.y)
                    zs.append(pos.z)

    mean_x = np.mean(np.array(xs))
    mean_y = np.mean(np.array(ys))
    mean_z = np.mean(np.array(zs))

    return mean_x, mean_y, mean_z


def center_structure(structure, point):
    mean_x, mean_y, mean_z = get_structure_mean(structure)

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    new_x = pos.x - mean_x + point[0]
                    new_y = pos.y - mean_y + point[1]
                    new_z = pos.z - mean_z + point[2]
                    atom.pos = gemmi.Position(new_x, new_y, new_z)

    return structure


def get_probe_structure(
        structure,
        bond_probes=False,
        neighbourhood_probes=False
):
    structure_clone = structure.clone()
    structure_clone.remove_hydrogens()

    j = 0
    verticies = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom_1 in residue:
                    # print(atom_1.element.name)
                    if atom_1.element.name != "H":
                        verticies[j] = atom_1
                        j = j + 1
    if bond_probes:
        edges = {}
        for atom_1_index in verticies:
            for atom_2_index in verticies:
                if atom_1_index == atom_2_index:
                    continue
                atom_1 = verticies[atom_1_index]
                atom_2 = verticies[atom_2_index]
                pos_1 = atom_1.pos
                pos_2 = atom_2.pos
                distance = pos_1.dist(pos_2)
                if distance < 2.0:
                    virtual_atom = gemmi.Atom()
                    new_pos = gemmi.Position(
                        (pos_1.x + pos_2.x) / 2,
                        (pos_1.y + pos_2.y) / 2,
                        (pos_1.z + pos_2.z) / 2,

                    )
                    atom_symbol: str = "C"
                    virtual_atom.name = atom_symbol
                    gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)
                    virtual_atom.element = gemmi_element
                    virtual_atom.pos = new_pos

                    if atom_1_index < atom_2_index:
                        edges[(atom_1_index, atom_2_index)] = virtual_atom

                    else:
                        edges[(atom_2_index, atom_1_index)] = virtual_atom

        for model in structure_clone:
            for chain in model:
                for residue in chain:
                    for edge_index in edges:
                        virtual_atom = edges[edge_index]
                        residue.add_atom(virtual_atom)

    if neighbourhood_probes:
        neighbourhood_probe_atoms = []
        for atom_index in verticies:

            atom = verticies[atom_index]
            pos = atom.pos
            for dx, dy, dz in itertools.product(
                    [-0.4, 0.4],
                    [-0.4, 0.4],
                    [-0.4, 0.4],
            ):
                virtual_atom = gemmi.Atom()
                new_pos = gemmi.Position(
                    pos.x + dx,
                    pos.y + dy,
                    pos.z + dz,

                )
                atom_symbol: str = "C"
                virtual_atom.name = atom_symbol
                gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)
                virtual_atom.element = gemmi_element
                virtual_atom.pos = new_pos
                neighbourhood_probe_atoms.append(virtual_atom)

        for model in structure_clone:
            for chain in model:
                for residue in chain:
                    for virtual_atom in neighbourhood_probe_atoms:
                        residue.add_atom(virtual_atom)
    # print(f"Number of real atoms: {len(verticies)}")
    # print(f"Number of virtual atoms: {len(edges)}")

    return structure_clone

def get_negative_probe_structure(
        structure,
        structure_array
):
    structure_clone = structure.clone()
    del structure_clone[0]
    model: gemmi.Model = gemmi.Model(f"0")
    structure_clone.add_model(model)
    chain: gemmi.Chain = gemmi.Chain(f"A")
    structure_clone[0].add_chain(chain)
    residue: gemmi.Residue = gemmi.Residue()
    residue.name = "LIG"
    residue.seqid = gemmi.SeqId(1, ' ')
    structure_clone[0][0].add_residue(residue)


    neighbourhood_probe_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom_1.element.name)
                    if atom.element.name != "H":
                        pos = atom.pos
                        for dx, dy, dz in itertools.product(
                                [-1.5, 1.5],
                                [-1.5, 1.5],
                                [-1.5, 1.5],
                        ):
                            deltas = structure_array - np.array([pos.x+dx, pos.y+dy, pos.z+dz]).reshape((1, 3))
                            dists = np.linalg.norm(deltas, axis=1)
                            if np.any(dists < 1.4):
                                continue

                            virtual_atom = gemmi.Atom()
                            new_pos = gemmi.Position(
                                pos.x + dx,
                                pos.y + dy,
                                pos.z + dz,

                            )
                            atom_symbol: str = "C"
                            virtual_atom.name = atom_symbol
                            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)
                            virtual_atom.element = gemmi_element
                            virtual_atom.pos = new_pos
                            neighbourhood_probe_atoms.append(virtual_atom)

    for model in structure_clone:
        for chain in model:
            for residue in chain:
                for virtual_atom in neighbourhood_probe_atoms:
                    residue.add_atom(virtual_atom)

    return structure_clone

def transform_structure_array(
        structure_array,
        transform_array,
        rotation_matrix,
):
    structure_mean = np.mean(structure_array, axis=0)

    demeaned_structure = structure_array - structure_mean

    rotated_structure = np.matmul(demeaned_structure, rotation_matrix)

    transformed_array = rotated_structure + structure_mean + transform_array

    return transformed_array


def get_interpolated_values_c(
        grid,
        transformed_structure_array,
        n,
):
    # vals = np.zeros(n, dtype=np.float32)

    # vals_list = \
    vals = grid.interpolate_position_array(
        transformed_structure_array.astype(np.float64),

    )
    # print(f"Vals list: {vals_list}")
    # print(f"Vals: {vals}")

    return vals


def score_fit_nonquant_array(structure_array, grid, distance, params):
    x, y, z, rx, ry, rz = params

    x_2 = distance * x
    y_2 = distance * y
    z_2 = distance * z

    rotation = spsp.transform.Rotation.from_euler(
        "xyz",
        [
            rx * 360,
            ry * 360,
            rz * 360,
        ],
        degrees=True,
    )
    rotation_matrix: np.ndarray = rotation.as_matrix()

    transformed_structure_array = transform_structure_array(
        structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )

    n = transformed_structure_array.shape[0]

    vals = get_interpolated_values_c(grid, transformed_structure_array, n)

    vals[vals > 3.0] = 3.0
    # vals[vals < 0.0] = 0.0

    score = np.sum(vals)

    return float(-score)

def score_fit_diff_array(structure_array, negative_structure_array, grid, distance, params):
    x, y, z, rx, ry, rz = params

    x_2 = distance * x
    y_2 = distance * y
    z_2 = distance * z

    rotation = spsp.transform.Rotation.from_euler(
        "xyz",
        [
            rx * 360,
            ry * 360,
            rz * 360,
        ],
        degrees=True,
    )
    rotation_matrix: np.ndarray = rotation.as_matrix()

    transformed_structure_array = transform_structure_array(
        structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )

    n = transformed_structure_array.shape[0]

    vals = get_interpolated_values_c(grid, transformed_structure_array, n)

    vals[vals > 2.0] = 2.0
    vals[vals < -2.0] = -2.0

    # Negative structure
    transformed_negative_structure_array = transform_structure_array(
        negative_structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )
    n = transformed_negative_structure_array.shape[0]

    negative_vals = get_interpolated_values_c(grid, transformed_negative_structure_array, n)

    negative_vals[negative_vals < -2.0] = -2.0
    negative_vals[negative_vals > 2.0] = 2.0


    # Score the conformor
    # Score on the percentage of outer mask values beneath the 15th percentile of inner mask values,
    # and penalize for the fraction of inner mask vals less than 0
    q = np.quantile(vals, 0.15)
    # q_neg = np.quantile(vals, 0.85)
    score = (np.sum(negative_vals <= q) / negative_vals.size) - (np.sum(vals <0.0) / vals.size) #* (np.sum(positive_vals))

    return float(-score)

def score_fit_mask_diff_array(structure_array, negative_structure_array, grid, distance, params):
    x, y, z, rx, ry, rz = params

    x_2 = distance * x
    y_2 = distance * y
    z_2 = distance * z

    rotation = spsp.transform.Rotation.from_euler(
        "xyz",
        [
            rx * 360,
            ry * 360,
            rz * 360,
        ],
        degrees=True,
    )
    rotation_matrix: np.ndarray = rotation.as_matrix()

    transformed_structure_array = transform_structure_array(
        structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )

    n = transformed_structure_array.shape[0]

    vals = get_interpolated_values_c(grid, transformed_structure_array, n)

    vals[vals > 2.0] = 2.0
    vals[vals < -2.0] = -2.0

    # Negative structure
    transformed_negative_structure_array = transform_structure_array(
        negative_structure_array,
        np.array([x_2, y_2, z_2]),
        rotation_matrix
    )
    n = transformed_negative_structure_array.shape[0]

    negative_vals = get_interpolated_values_c(grid, transformed_negative_structure_array, n)

    negative_vals[negative_vals < -2.0] = -2.0
    negative_vals[negative_vals > 2.0] = 2.0


    # Score the conformor
    # Score on the percentage of outer mask values beneath the 15th percentile of inner mask values,
    # and penalize for the fraction of inner mask vals less than 0
    q = np.quantile(vals, 0.5)
    # q_neg = np.quantile(vals, 0.85)
    score = ((np.sum(vals >= 0.5) / vals.size) - (np.sum(negative_vals >= 0.5) / negative_vals.size)) - (np.sum(vals <0.5) / vals.size) #* (np.sum(positive_vals))

    return float(-score)


def transform_structure(structure, translation, rotation_matrix):
    mean_x, mean_y, mean_z = get_structure_mean(structure)
    structure_copy = structure.clone()
    structure_copy = center_structure(structure_copy, [0.0, 0.0, 0.0])

    transform: gemmi.Transform = gemmi.Transform()
    transform.mat.fromlist(rotation_matrix.tolist())
    transform.vec.fromlist([0.0, 0.0, 0.0])

    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    rotated_vec = transform.apply(pos)
                    rotated_position = gemmi.Position(rotated_vec.x, rotated_vec.y, rotated_vec.z)
                    atom.pos = rotated_position

    structure_copy = center_structure(
        structure_copy,
        [
            mean_x + translation[0],
            mean_y + translation[1],
            mean_z + translation[2]
        ]
    )

    return structure_copy


def _neg_shell_offsets(hv_coords):
    """Build the 'negative probe' halo: points ~1.5 A outside each heavy atom,
    excluding any within 1.4 A of a real atom (cf get_negative_probe_structure).
    This is the discreteness/anti-bulk term -- density should sit ON the ligand
    and be empty just AROUND it, so a pose that wanders into bulk (protein) is
    penalised."""
    offs = list(itertools.product([-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]))
    pts = []
    for p in hv_coords:
        for d in offs:
            q = p + np.array(d, dtype=np.float64)
            if np.min(np.linalg.norm(hv_coords - q, axis=1)) < 1.4:
                continue
            pts.append(q)
    return np.array(pts, dtype=np.float64) if pts else hv_coords.copy()


def _refine_pose_on_density(seed, de_grid, max_shift=3.0, maxfev=60):
    """Local refine of a seed pose on DE's *anchored* objective: maximise ligand
    density on the (protein-masked) score grid while keeping the surrounding
    shell off-density. Rotation (rotvec about the heavy-atom centroid) + a
    translation HARD-BOUNDED to +/-max_shift of the seed. Because the grid is
    protein-masked (no reward on protein) and the shell penalises bulk density --
    and the FRF already centred the seed on the event -- this cannot drift onto
    the protein the way a CNN-driven refine does. Returns the refined structure.
    """
    atoms, base = [], []
    for model in seed:
        for chain in model:
            for r in chain:
                for a in r:
                    atoms.append(a)
                    base.append([a.pos.x, a.pos.y, a.pos.z])
    base = np.asarray(base, dtype=np.float64)
    heavy = np.array([base[i] for i, a in enumerate(atoms)
                      if a.element.name != "H"], dtype=np.float64)
    c0 = heavy.mean(axis=0)
    shell = _neg_shell_offsets(heavy)

    def _xform(pts, p):
        rot = spsp.transform.Rotation.from_rotvec(p[:3]).as_matrix()
        return (pts - c0) @ rot.T + c0 + p[3:]

    def _interp(pts):
        return np.array([de_grid.interpolate_value(
            gemmi.Position(float(x), float(y), float(z))) for x, y, z in pts])

    def _neg(p):
        if np.linalg.norm(p[3:]) > max_shift:   # hard translation bound
            return 10.0
        lv = _interp(_xform(heavy, p))
        sv = _interp(_xform(shell, p))
        # DE's score: ligand-on-density - shell-on-density - ligand-off-density.
        score = (np.mean(lv >= 0.5) - np.mean(sv >= 0.5) - np.mean(lv < 0.5))
        return -float(score)

    step = np.array([0.15, 0.15, 0.15, 0.5, 0.5, 0.5])
    simplex = np.vstack([np.zeros(6)] + [np.eye(6)[i] * step[i] for i in range(6)])
    r = optimize.minimize(
        _neg, np.zeros(6), method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-2, "fatol": 1e-3,
                 "maxfev": maxfev})
    rot = spsp.transform.Rotation.from_rotvec(r.x[:3]).as_matrix()
    nc = (base - c0) @ rot.T + c0 + r.x[3:]
    st = seed.clone()
    sa = [a for model in st for chain in model for rr in chain for a in rr]
    for a, c in zip(sa, nc):
        a.pos = gemmi.Position(float(c[0]), float(c[1]), float(c[2]))
    return st


def _score_conformer_crowther(centroid_cart, conformer, score_build, z_grid,
                         raw_xmap_grid, res=None, seed_target=None,
                         de_grid=None, n_seeds=10):
    """FRF-seeded minimise-then-score pose search -- the DE replacement.

    Same shape as the DE path (refine candidate poses on an anchored density
    objective, then CNN-rank the results), but seeded by the SH-Crowther FRF
    instead of DE's random restarts -- informed seeding rather than a global
    search. For each FRF seed we locally refine on DE's own objective
    (``_refine_pose_on_density``: protein-masked grid + negative shell, bounded
    translation) and then let the build CNN *rank* the refined poses. The CNN
    only selects; it never steers the pose (steering it drifts the ligand onto
    protein density, since the build CNN scores the raw unmasked map in a box
    that re-centres on the ligand). The masked grid + shell + translation bound
    keep every pose anchored on the event.

    ``seed_target`` is the FRF target (protein-masked 1-BDC event map);
    ``de_grid`` is the anchored refine target (the DE score grid); ``z_grid`` +
    ``raw_xmap_grid`` are the unmasked maps the CNN ranks on. Returns
    (structure, cnn, centroid, arr).
    """
    from .crowther.fit import (
        CrowtherConfig, get_precompute, prepare_event_target,
        fit_conformer_against, sigma_from_resolution)

    coords = np.array(
        [[a.pos.x, a.pos.y, a.pos.z]
         for model in conformer for chain in model for res in chain
         for a in res if a.element.name != "H"],
        dtype=np.float64,
    )
    ligand_radius = float(np.linalg.norm(
        coords - coords.mean(axis=0), axis=1).max()) + 2.0

    # Tunable without recompiling.
    n_seeds = int(os.environ.get("PANDDA_CROWTHER_NSEEDS", n_seeds))
    max_shift = float(os.environ.get("PANDDA_CROWTHER_MAX_SHIFT", 3.0))

    sigma = sigma_from_resolution(res) if res is not None else None
    pre = get_precompute(CrowtherConfig())
    target = prepare_event_target(
        z_grid if seed_target is None else seed_target,
        centroid_cart, pre, ligand_radius=ligand_radius)
    candidates = fit_conformer_against(
        target, conformer, pre, sigma=sigma, n_candidates=n_seeds)

    # minimise (anchored DE objective) then score (CNN rank).
    best = None
    for struct, _tani, _cen in candidates:
        refined = (_refine_pose_on_density(struct, de_grid, max_shift=max_shift)
                   if de_grid is not None else struct)
        sc, arr = score_build(refined, z_grid, raw_xmap_grid)
        sc = float(np.ravel(sc)[0])
        if best is None or sc > best[1]:
            best = (refined, sc, arr)
    struct, score, arr = best
    cen = get_structure_mean(struct)
    # Safety gate: the refine is translation-bounded, so a build far from the
    # event signals a bug (the drift we fixed). Log it loudly.
    d = float(np.linalg.norm(np.asarray(cen, float) - np.asarray(centroid_cart, float)))
    if os.environ.get("PANDDA_CROWTHER_DBG"):
        seed_d = sorted(float(np.linalg.norm(np.asarray(c[2], float) - np.asarray(centroid_cart, float)))
                        for c in candidates)
        print(f"CROWTHERDBG3 centroid_cart={np.round(np.asarray(centroid_cart,float),1)} "
              f"FRF_seed_dists={[round(x,1) for x in seed_d]} build_dist={d:.1f}", flush=True)
    if d > 6.0:
        print(f"WARNING: crowther build {d:.1f} A from event centroid "
              f"(translation bound {max_shift} A -- should be < ~{max_shift + 2:.0f} A)")
    return (struct, score, cen, arr)


def score_conformer(
        centroid_cart,
        conformer,
        zmap_grid,
            score_build,
            z_grid,
            raw_xmap_grid,
        #event_fit_num_trys=6,
        event_fit_num_trys=12,
        res=None,
):
    # Experimental SH-Crowther FRF pose search (A/B against the DE search below).
    # Toggle with PANDDA_CROWTHER_FIT=1; env flag keeps the switch out of the call
    # chain for the prototype. Uses the z map as the FRF target (HOLE 1) and
    # keeps score_build (the CNN) as the score arbiter, so the return contract
    # and everything downstream are unchanged.
    if os.environ.get("PANDDA_CROWTHER_FIT"):
        # FRF seeds on the z-map (cut_cube carves its own 64^3 orthonormal cube,
        # so the source grid's space group/geometry is irrelevant), then refines
        # each seed on the DE score grid (zmap_grid: masked + anchored, bounded
        # translation) and CNN-ranks. Same anchored minimise-then-score as the
        # local path, but on full grids -> no cut_local_grid_from_sparse fold bug.
        return _score_conformer_crowther(
            centroid_cart, conformer, score_build, z_grid, raw_xmap_grid, res,
            seed_target=z_grid, de_grid=zmap_grid)

    centered_structure = center_structure(
        conformer,
        centroid_cart,
    )

    # Get the probe structure
    probe_structure = get_probe_structure(centered_structure)

    # Get structure array
    structure_positions = []

    for model in probe_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        pos = atom.pos
                        structure_positions.append([pos.x, pos.y, pos.z])

    structure_array = np.array(structure_positions, dtype=np.float32)

    # Get the negative probe
    negative_probe_structure = get_negative_probe_structure(
        centered_structure,
        structure_array
    )

    negative_structure_positions = []
    for model in negative_probe_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != "H":
                        pos = atom.pos
                        negative_structure_positions.append([pos.x, pos.y, pos.z])
    negative_structure_array = np.array(negative_structure_positions, dtype=np.float32)

    # print(f'Probe shape: {structure_array.shape} {np.mean(structure_array, axis=0)}')
    # print(f'Negative Probe shape: {negative_structure_array.shape} {np.mean(negative_structure_array, axis=0)}')

    # Optimise



    scores = []
    optimised_structures = []

    total_evolve_time = 0.0
    time_begin_score = time.time()
    for j in range(event_fit_num_trys):
        # print(f"\t\t\t\tOptimizing round {j}")
        time_begin_evolve = time.time()
        res = optimize.differential_evolution(
            # lambda params: score_fit_nonquant_array(
            #     structure_array,
            #     zmap_grid,
            #     1.0,
            #     params
            # ),
            lambda params: score_fit_mask_diff_array(
                structure_array,
                negative_structure_array,
                zmap_grid,
                1.0,
                params
            ),
            [
                (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0),
                (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            ],
            seed=_de_seed(),
            # popsize=30,
        )
        # res = optimize.shgo(
        #     lambda params: score_fit_nonquant_array(
        #         structure_array,
        #         zmap_grid,
        #         1.0,
        #         params
        #     ),
        #     [
        #         (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        #     ],
        #     # sampling_method='sobol',
        #     # n=10000,
        #     iters=3
        #     # popsize=30,
        # )
        # res = optimize.basinhopping(
        #     lambda params: score_fit_nonquant_array(
        #         structure_array,
        #         zmap_grid,
        #         1.0,
        #         params
        #     ),
        #     [0.0,0.0,0.0,0.5,0.5,0.5],
        #     stepsize=0.05,
        #     niter=200
        #     # [
        #     #     (-6.0, 6.0), (-6, 6.0), (-6.0, 6.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        #     # ],
        #     # sampling_method='sobol',
        #     # n=10000,
        #     # iters=3
        #     # popsize=30,
        # )
        # print(res)
        time_finish_evolve = time.time()
        total_evolve_time += (time_finish_evolve - time_begin_evolve)
        # print(f"\t\t\t\t\tFinished Optimizing round {j}")



        # Get optimised fit
        x, y, z, rx, ry, rz = res.x
        rotation = spsp.transform.Rotation.from_euler(
            "xyz",
            [
                rx * 360,
                ry * 360,
                rz * 360,
            ],
            degrees=True)
        rotation_matrix: np.ndarray = rotation.as_matrix().T

        optimised_structure = transform_structure(
            centered_structure,
            [x, y, z],
            rotation_matrix
        )
        optimised_structures.append(optimised_structure)

        score, arr = score_build(
            optimised_structure,
            # reference_frame.unmask(SparseDMap(unmasked_dtag_array)),
            z_grid,
            raw_xmap_grid,
        )

        # scores.append(res.fun)
        scores.append(-score)

        centroid = get_structure_mean(optimised_structure)
        # print(f'Optimization {j} Score {round(float(res.fun), 3)} Centroid {round(centroid[0], 2)} {round(centroid[1], 2)} {round(centroid[2], 2)}')
    time_finish_score = time.time()
    # print(f"\t\t\tScored conformer in {time_finish_score-time_begin_score} seconds, of which {total_evolve_time} evolving!")

    # print(f"Scores: {scores}")
    best_score_index = np.argmin(scores)

    best_score_fit_score = scores[best_score_index]
    best_optimised_structure = optimised_structures[best_score_index]

    return best_optimised_structure, float(-best_score_fit_score), get_structure_mean(best_optimised_structure), arr


def get_score_grid(dmap, st, event: EventInterface):
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    ns = gemmi.NeighborSearch(st.structure[0], st.structure.cell, 12).populate(include_h=False)

    centroid = np.mean(event.pos_array, axis=0)

    centoid_pos = gemmi.Position(*centroid)
    marks = ns.find_atoms(centoid_pos, '\0', radius=11)

    for mark in marks:
        cra = mark.to_cra(st.structure[0])
        residue = cra.residue
        if residue.name in constants.RESIDUE_NAMES:
            # mark_pos = mark.pos
            # pos = gemmi.Position(mark_pos.x, mark_pos.y, mark_pos.z)
            pos = gemmi.Position(mark.x, mark.y, mark.z)
            inner_mask_grid.set_points_around(
                pos,
                radius=1.5,
                value=1,
            )
    # #
    # for model in st.structure:
    #     for chain in model:
    #         for residue in chain:
    #             if residue.name in constants.RESIDUE_NAMES:
    #                 for atom in residue:
    #                     pos = atom.pos
    #                     inner_mask_grid.set_points_around(pos,
    #                                                       radius=1.5,
    #                                                       value=1,
    #                                                       )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
    dmap_array[structure_mask_indicies] = 0.0

    return dmap


def mask_dmap(dmap_array, st, reference_frame):
    dmap = reference_frame.unmask(SparseDMap(dmap_array))
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name in constants.RESIDUE_NAMES:
                    for atom in residue:
                        pos = atom.pos
                        inner_mask_grid.set_points_around(pos,
                                                          radius=1.5,
                                                          value=1,
                                                          )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
    dmap_array[structure_mask_indicies] = 0.0

    return SparseDMap.from_xmap(dmap, reference_frame).data


def get_event_grid(dmap, st, ):
    # Get a mask of the protein
    inner_mask_grid = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(dmap.unit_cell)

    # ns = gemmi.NeighborSearch(st.structure[0], st.structure.cell, 12).populate(include_h=False)

    # centroid = np.mean(event.pos_array, axis=0)
    #
    # centoid_pos = gemmi.Position(*centroid)
    # marks = ns.find_atoms(centoid_pos, '\0', radius=11)

    # for mark in marks:
    #     cra = mark.to_cra(st.structure[0])
    #     residue = cra.residue
    #     if residue.name in constants.RESIDUE_NAMES:
    #         # mark_pos = mark.pos
    #         # pos = gemmi.Position(mark_pos.x, mark_pos.y, mark_pos.z)
    #         pos = gemmi.Position(mark.x, mark.y, mark.z)
    #         inner_mask_grid.set_points_around(
    #             pos,
    #             radius=1.5,
    #             value=1,
    #         )
    #
    for model in st.structure:
        for chain in model:
            for residue in chain:
                if residue.name in constants.RESIDUE_NAMES:
                    for atom in residue:
                        pos = atom.pos
                        inner_mask_grid.set_points_around(pos,
                                                          radius=1.5,
                                                          value=1,
                                                          )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)
    # print(inner_mask_grid_array.size)

    # Zero out density overlapping the protein
    dmap_array = np.array(dmap, copy=False)
    # non_zero_dmap_array = d
    # print(f"")
    structure_mask_indicies = np.nonzero(inner_mask_grid_array)
    # print(f"Mask indicies size: {inner_mask_grid_array[0].size}")
    dmap_array[structure_mask_indicies] = 0.0

    return dmap


class AutobuildInbuilt:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(
            self,
            event: EventInterface,
            dataset: DatasetInterface,
            dmap_path,
            mtz_path,
            model_path,
            ligand_files,
            out_dir,
    ):

        # Get the structure
        # st = Structure.from_path(model_path)
        # st = dataset.structure
        st = Structure.from_path(str(dataset.structure.path))

        # Get the scoring grid
        dmap = load_dmap(dmap_path)
        score_grid = get_score_grid(dmap, st, event)

        # save_dmap(score_grid, out_dir / "score_grid.ccp4")

        # ligand_scoring_results = {}
        # for ligand_key, ligand_files in dataset.ligand_files.items():

        # Generate conformers to score
        # print(f"\tGetting conformermers!")
        conformers = get_conformers(ligand_files)
        # print(f"\t\tGot {len(conformers)} conformers!")

        if len(conformers) == 0:
            return

        # Score conformers against the grid
        conformer_scores = {}
        for conformer_id, conformer in conformers.items():
            optimized_structure, score, centroid = score_conformer(
                np.mean(event.pos_array, axis=0),
                conformer,
                score_grid,
            )
            conformer_scores[conformer_id] = [optimized_structure, score]
            # print(f"\tLigand: {ligand_key}: Conformer: {conformer_id}: Score: {score}")

        # ligand_scoring_results[ligand_key] = conformer_scores

        if len(conformer_scores) == 0:
            return AutobuildResult(
                {},
                dmap_path,
                mtz_path,
                model_path,
                ligand_files.ligand_cif,
                out_dir
            )

        # Choose the best ligand
        # if len(ligand_scoring_results) == 0:
        #     return AutobuildResult(
        #         {},
        #         dmap_path,
        #         mtz_path,
        #         model_path,
        #         cif_path,
        #         out_dir
        #     )

        # best_ligand_key = max(
        #     ligand_scoring_results,
        #     key=lambda _ligand_key: max(
        #         ligand_scoring_results[_ligand_key],
        #         key=lambda _conformer_id: ligand_scoring_results[_ligand_key][_conformer_id][1],
        #     )
        # )
        #
        # best_ligand_conformer_scores = ligand_scoring_results[best_ligand_key]

        # Save the fit conformers
        # for conformer_id, (optimized_structure, score) in best_ligand_conformer_scores.items():
        #     save_structure(
        #         Structure(None, optimized_structure),
        #         out_dir / f"{conformer_id}.pdb",
        #     )
        #
        # log_result_dict = {
        #     str(out_dir / f"{conformer_id}.pdb"): score
        #     for conformer_id, (optimized_structure, score)
        #     in best_ligand_conformer_scores.items()
        # }
        for conformer_id, (optimized_structure, score) in conformer_scores.items():
            save_structure(
                Structure(None, optimized_structure),
                out_dir / f"{conformer_id}.pdb",
            )

        log_result_dict = {
            str(out_dir / f"{conformer_id}.pdb"): score
            for conformer_id, (optimized_structure, score)
            in conformer_scores.items()
        }

        # Return results
        return AutobuildResult(
            log_result_dict,
            dmap_path,
            mtz_path,
            model_path,
            ligand_files.ligand_cif,
            out_dir
        )


def get_local_signal_dencalc(optimized_structure, event_map_grid, res, ):
    # Get the electron density of the optimized structure
    optimized_structure.cell = event_map_grid.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = res  # *2
    dencalc.rate = 2.0
    # initial_dencalc_grid = gemmi.FloatGrid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    # initial_dencalc_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # initial_dencalc_grid.set_unit_cell(event_map_grid.unit_cell)
    # dencalc.grid = initial_dencalc_grid
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    dencalc.put_model_density_on_grid(optimized_structure[0])
    # dencalc.add_model_density_to_grid(optimized_structure[0])
    calc_grid = dencalc.grid
    calc_grid_array = np.array(calc_grid, copy=False)
    # print([event_map_grid.nu, event_map_grid.nv, event_map_grid.nw, calc_grid.nu, calc_grid.nv, calc_grid.nw])
    # print([calc_grid.nu, event_map_grid.nu])

    # Get the mask around the structure
    inner_mask_grid = gemmi.Int8Grid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(event_map_grid.unit_cell)

    num_atoms = 0
    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                # if residue.name in constants.RESIDUE_NAMES:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )
                    num_atoms += 1

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)

    # Get the correlation with the event
    event_map_grid_array = np.array(event_map_grid, copy=False)
    masked_event_map_vals = event_map_grid_array[inner_mask_grid_array >= 2]
    masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]
    corr = np.corrcoef(
        np.concatenate(
            (
                masked_event_map_vals.reshape(-1, 1),
                masked_calc_vals.reshape(-1, 1)
            ),
            axis=1,
        )
    )[0, 1]

    num_atoms = np.log(num_atoms)

    return corr  # * num_atoms


def get_correlation(_bdc, masked_xmap_vals, masked_mean_map_vals, masked_calc_vals):
    event_map_vals = (masked_xmap_vals - (_bdc * masked_mean_map_vals)) / (1 - _bdc)
    masked_event_map_vals = event_map_vals[event_map_vals != 0]
    double_masked_calc_vals = masked_calc_vals[event_map_vals != 0]

    if masked_event_map_vals.size == 0:
        # print(f'No overlap between event map vals != 0 and masked event map vals!')
        return 1.0

    # corr = np.corrcoef(
    #     np.concatenate(
    #         (
    #             masked_event_map_vals.reshape(-1, 1),
    #             masked_calc_vals.reshape(-1, 1)
    #         ),
    #         axis=1,
    #     )
    # )[0, 1]
    # return 1-corr

    event_map_mean = np.mean(masked_event_map_vals)
    calc_map_mean = np.mean(double_masked_calc_vals)
    delta_event_map = masked_event_map_vals - event_map_mean
    delta_calc_map = double_masked_calc_vals - calc_map_mean
    nominator = np.sum(delta_event_map * delta_calc_map)
    denominator = np.sqrt(
        np.sum(np.square(delta_event_map)) * np.sum(np.square(delta_calc_map))
    )

    corr = nominator / denominator
    # print(f'BDC: {round(_bdc, 2)} : Correlation: {corr}')
    return 1 - corr


def get_predicted_mask(
        optimized_structure,
        xmap,
):
    # Get the mask around the structure
    inner_mask_grid = gemmi.Int8Grid(xmap.nu, xmap.nv, xmap.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(xmap.unit_cell)

    num_atoms = 0
    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.5,
                                                      value=4,
                                                      )
                    num_atoms += 1

    return inner_mask_grid


def get_predicted_density(
        optimized_structure,
        xmap
):
    # Get the electron density of the optimized structure
    optimized_structure.cell = xmap.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    # dencalc.d_min = res#*2
    # dencalc.rate = 2.0
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    # dencalc.initialize_grid_to_size(xmap.nu, xmap.nv, xmap.nw)
    dencalc.grid.set_size(xmap.nu, xmap.nv, xmap.nw)

    dencalc.add_model_density_to_grid(optimized_structure[0])
    # dencalc.put_model_density_on_grid(optimized_structure[0])
    calc_grid = dencalc.grid
    # calc_grid_array = np.array(calc_grid, copy=False)

    return calc_grid


def get_predicted_density_high_contour(
        predicted_density,
        predicted_mask
):
    predicted_density_array = np.array(predicted_density, copy=False)
    predicted_mask_array = np.array(predicted_mask, copy=False)

    predicted_high_vals = predicted_density_array[predicted_mask_array > 3]

    contour = np.quantile(
        predicted_high_vals,
        0.0
    )

    return contour


def get_local_signal_dencalc_optimize_bdc(
        optimized_structure,
        calc_grid,
        predicted_mask,
        reference_frame,
        dtag_vals,
        mean_vals,
        res,
        event_bdc,
):
    # Get the unmasked xmap and mean map
    xmap = reference_frame.unmask(dtag_vals)
    mean_map = reference_frame.unmask(mean_vals)
    xmap_array = np.array(xmap, copy=False)
    mean_map_array = np.array(mean_map, copy=False)

    calc_grid_array = np.array(calc_grid, copy=False)

    inner_mask_grid_array = np.array(predicted_mask, copy=False)

    # Pull out the ligand masked xmap and mean map vals
    masked_xmap_vals = xmap_array[inner_mask_grid_array >= 2]
    masked_mean_map_vals = mean_map_array[inner_mask_grid_array >= 2]
    masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]
    # print(f'masked_xmap_vals: {masked_xmap_vals}')
    # print(f'masked_mean_map_vals: {masked_mean_map_vals}')
    # print(f'masked_calc_vals: {masked_calc_vals}')

    # res = optimize.minimize(
    #     lambda _bdc: get_correlation(_bdc, masked_xmap_vals, masked_mean_map_vals, masked_calc_vals),
    #     event_bdc,
    #     bounds=((0.0, 0.95),),
    #     # tol=0.1
    # )

    # print(f'Correlations')
    # print(get_correlation(0.0, masked_xmap_vals,
    #                       masked_mean_map_vals,
    #                       masked_calc_vals, ))
    # print(get_correlation(0.5, masked_xmap_vals,
    #                       masked_mean_map_vals,
    #                       masked_calc_vals, ))

    res = optimize.differential_evolution(
        lambda _bdc: get_correlation(
            _bdc,
            masked_xmap_vals,
            masked_mean_map_vals,
            masked_calc_vals,
        ),
        [(0.0, 0.95), ],
        seed=_de_seed(),
        # popsize=30,
    )

    # # Get the correlation with the event
    # event_map_grid_array = np.array(event_map_grid, copy=False)
    # masked_event_map_vals = event_map_grid_array[inner_mask_grid_array >= 2]
    # masked_calc_vals = calc_grid_array[inner_mask_grid_array >= 2]
    # corr = np.corrcoef(
    #     np.concatenate(
    #         (
    #             masked_event_map_vals.reshape(-1,1),
    #             masked_calc_vals.reshape(-1, 1)
    #         ),
    #         axis=1,
    #     )
    # )[0,1]

    # num_atoms = np.log(num_atoms)
    bdc = res.x
    corr = 1 - res.fun

    # masked_event_map_vals = (masked_xmap_vals - (bdc * masked_mean_map_vals)) / (1 - bdc)

    # print(f"Refined to bdc: {bdc} and correlation: {corr} on set of size: {masked_xmap_vals.size}")

    return corr, bdc  # * num_atoms


def get_local_signal(optimized_structure, event_map_grid):
    event_map_grid_array = np.array(event_map_grid, copy=False)

    event_map_grid_array[event_map_grid_array < 0.0] = 0.0
    event_map_grid_array[event_map_grid_array > 2.0] = 2.0
    inner_mask_grid = gemmi.Int8Grid(event_map_grid.nu, event_map_grid.nv, event_map_grid.nw)
    inner_mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    inner_mask_grid.set_unit_cell(event_map_grid.unit_cell)

    for model in optimized_structure:
        for chain in model:
            for residue in chain:
                # if residue.name in constants.RESIDUE_NAMES:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos = atom.pos
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.5,
                                                      value=1,
                                                      )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=1.0,
                                                      value=2,
                                                      )
                    # inner_mask_grid.set_points_around(pos,
                    #                                   radius=1.0,
                    #                                   value=2,
                    #                                   )
                    inner_mask_grid.set_points_around(pos,
                                                      radius=0.75,
                                                      value=3,
                                                      )

    inner_mask_grid_array = np.array(inner_mask_grid, copy=False)

    # vals_pos = event_map_grid_array[np.nonzero(inner_mask_grid_array == 2)]
    # vals_neg = event_map_grid_array[np.nonzero(inner_mask_grid_array == 1)]
    full_mask = event_map_grid_array[np.nonzero(inner_mask_grid_array >= 1)]
    background = np.mean(full_mask)

    outer_mask = event_map_grid_array[np.nonzero(inner_mask_grid_array == 1)]
    outer_mean = np.mean(outer_mask)
    outer_std = np.std(outer_mean)
    # background = np.mean(outer_mask)
    background = outer_mean + (2 * outer_std)

    # high_non_core = np.sum(outer_mask > background)
    # low_non_core = np.sum(outer_mask <= background)
    # non_core_score = low_non_core-high_non_core
    # non_core_score = low_non_core * (low_non_core / outer_mask.size)

    core_points = event_map_grid_array[np.nonzero(inner_mask_grid_array == 3)]
    high_core = np.sum(core_points > background)
    # low_core = np.sum(core_points <= background)
    # core_score = high_core-low_core
    # core_score = high_core * (high_core/core_points.size)

    # score = core_score+(non_core_score*(core_points.size / outer_mask.size))
    score = high_core

    # return np.sum(vals_pos-np.mean(vals_neg)) #- np.sum(vals_neg)
    return score


def get_signal(
        xmap_array,
        xmap_mask
):
    return xmap_array[xmap_mask]


def get_optimal_signal_contour(
        signal_vals,
        noise_signal_vals
):
    diffs = {}
    for val in np.linspace(
            np.min(signal_vals),
            np.max(signal_vals),
            num=100
    ):
        difference = int(np.sum(signal_vals > val) - np.sum(noise_signal_vals > val))
        diffs[float(val)] = difference

    return max(diffs, key=lambda _key: diffs[_key])


def get_contacts(
        optimized_structure,
        st
):
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate(include_h=False)

    contacts = []
    for model in optimized_structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    # contacts = []
                    marks = ns.find_atoms(atom.pos, '\0', radius=3.2)
                    for mark in marks:
                        cra = mark.to_cra(st[0])
                        res = cra.residue
                        if res.name in constants.RESIDUE_NAMES:
                            contacts.append(res.seqid.num)
    num_contacts = len(set(contacts))
    return num_contacts
    ...


def _translate_structure(st, vec):
    """Return a clone of gemmi structure ``st`` shifted by Cartesian ``vec``."""
    out = st.clone()
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    for model in out:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    p = atom.pos
                    atom.pos = gemmi.Position(p.x + vx, p.y + vy, p.z + vz)
    return out


def _protein_masked_event_local(rawx_local, mean_local, bdc, structure_local,
                                n, spacing):
    """Build the FRF seed target in the local box: the (1-BDC) background-
    corrected event map (xmap - bdc*mean) with the protein decoy zeroed (density
    within 1.5 A of any standard-residue atom). On the BAZ2B test set this is the
    best seed target -- it isolates the ligand difference density from both bulk
    background and the protein, giving the cleanest seeds. ``structure_local`` is
    the dataset model already translated into the box-local frame (-box_origin)."""
    ev = np.array(rawx_local, copy=True) - float(bdc) * np.array(mean_local, copy=False)
    seed = gemmi.FloatGrid(n, n, n)
    cell = gemmi.UnitCell(n * spacing, n * spacing, n * spacing, 90.0, 90.0, 90.0)
    seed.set_unit_cell(cell)
    seed.spacegroup = gemmi.SpaceGroup("P 1")
    np.array(seed, copy=False)[:, :, :] = ev.astype(np.float32)

    mask = gemmi.Int8Grid(n, n, n)
    mask.set_unit_cell(cell)
    mask.spacegroup = gemmi.SpaceGroup("P 1")
    box = n * spacing
    for model in structure_local:
        for chain in model:
            for residue in chain:
                if residue.name in constants.RESIDUE_NAMES:
                    for atom in residue:
                        p = atom.pos
                        if (-1.5 <= p.x <= box + 1.5 and -1.5 <= p.y <= box + 1.5
                                and -1.5 <= p.z <= box + 1.5):
                            mask.set_points_around(p, radius=1.5, value=1)
    np.array(seed, copy=False)[np.nonzero(np.array(mask, copy=False))] = 0.0
    return seed


def _autobuild_conformer_local(
        centroid, event_bdc, conformer, masked_dtag_array, masked_mean_array,
        reference_frame, out_dir, conformer_id, res, structure,
        unmasked_dtag_array, unmasked_mean_array, z_array, raw_xmap_sparse,
        score_build, raw_xmap_array_ref, n=96, spacing=0.5):
    """Memory-light autobuild: cut local boxes from the sparse maps about the event
    centroid (no full-cell unmask), fit with crowther + score with the CNN/BDC/
    signal entirely in that local box, then map the pose back to the native frame.
    Mirrors autobuild_conformer's outputs; result is frame-invariant since all
    scores are translation-invariant."""
    from .crowther.local import cut_local_grid_from_sparse

    normalize_z = (z_array - np.mean(z_array)) / np.std(z_array)
    normalize_xmap = (masked_dtag_array - np.mean(masked_dtag_array)) / np.std(masked_dtag_array)
    # The DE fit's score-grid target (same construction as the full path), built
    # sparsely so it can be cut locally -- needed so DE+local works (the
    # behaviour-preserving control); the crowther fit ignores it.
    score_grid_sparse = np.zeros(normalize_z.shape, dtype=np.float32)
    score_grid_sparse[normalize_xmap > 1.5] = 0.5
    score_grid_sparse[normalize_z > 1.5] = 1.0

    # One local frame (box_origin is deterministic from centroid/n/spacing, so all
    # cuts share it).
    z_local, box_origin = cut_local_grid_from_sparse(reference_frame, normalize_z, centroid, n, spacing)
    event_local, _ = cut_local_grid_from_sparse(reference_frame, score_grid_sparse, centroid, n, spacing)
    rawx_local, _ = cut_local_grid_from_sparse(reference_frame, raw_xmap_sparse, centroid, n, spacing)
    xmap_local, _ = cut_local_grid_from_sparse(reference_frame, masked_dtag_array, centroid, n, spacing)
    dtag_local, _ = cut_local_grid_from_sparse(reference_frame, unmasked_dtag_array, centroid, n, spacing)
    mean_local, _ = cut_local_grid_from_sparse(reference_frame, unmasked_mean_array, centroid, n, spacing)

    centroid_local = np.asarray(centroid, dtype=np.float64) - box_origin
    conf_local = _translate_structure(conformer.structure, -box_origin)

    # crowther path: FRF seeds against the protein-masked (1-BDC) event map, then
    # each seed is refined directly on the CNN (z_local/rawx_local, unmasked). DE
    # path: differential_evolution against event_local (crowther ignores it).
    if os.environ.get("PANDDA_CROWTHER_DBG"):
        za = np.array(z_local, copy=False); ea = np.array(event_local, copy=False)
        zhi = np.argwhere(za > 2.0); ehi = np.argwhere(ea >= 0.5)
        zc = (zhi.mean(0) * spacing) if len(zhi) else None
        ec = (ehi.mean(0) * spacing) if len(ehi) else None
        print(f"CROWTHERDBG2 centroid_local={np.round(np.asarray(centroid_local,float),1)} "
              f"z_local_mass_centroid={np.round(zc,1) if zc is not None else None} "
              f"event_local_mass_centroid={np.round(ec,1) if ec is not None else None}",
              flush=True)
    if os.environ.get("PANDDA_CROWTHER_FIT"):
        # FRF seeds against z_local (the same centred grid the offline native
        # test placed at ~0.6 A); the reconstructed protein-masked event map was
        # off-centre and drove the seed ~14 A off the event. Refine on event_local.
        optimized_local, score, _cen, arr = _score_conformer_crowther(
            centroid_local, conf_local, score_build, z_local, rawx_local, res,
            seed_target=z_local, de_grid=event_local)
        if os.environ.get("PANDDA_CROWTHER_DBG"):
            bl = np.asarray(_cen, float)              # build centroid, LOCAL frame
            cl = np.asarray(centroid_local, float)    # event centroid, LOCAL frame
            print(f"CROWTHERDBG box_origin={np.round(box_origin,1)} "
                  f"event_local={np.round(cl,1)} build_local={np.round(bl,1)} "
                  f"build_local-event_local={np.linalg.norm(bl-cl):.1f}A "
                  f"build_native={np.round(bl+box_origin,1)} "
                  f"event_native={np.round(np.asarray(centroid,float),1)}", flush=True)
    else:
        optimized_local, score, _cen, arr = score_conformer(
            centroid_local, conf_local, event_local, score_build, z_local, rawx_local, res=res)

    predicted_mask = get_predicted_mask(optimized_local, xmap_local)
    predicted_mask_array = np.array(predicted_mask, copy=False)
    predicted_density = get_predicted_density(optimized_local, xmap_local)
    predicted_density_array = np.array(predicted_density, copy=False)
    try:
        high = get_predicted_density_high_contour(predicted_density, predicted_mask)
    except Exception:
        high = 1.0

    # BDC by maximising calc-vs-event correlation over the ligand mask, on local grids
    da = np.array(dtag_local, copy=False)
    me = np.array(mean_local, copy=False)
    sel = predicted_mask_array >= 2
    if int(sel.sum()) > 0:
        rr = optimize.differential_evolution(
            lambda b: get_correlation(b, da[sel], me[sel], predicted_density_array[sel]),
            [(0.0, 0.95)], seed=_de_seed())
        corr = 1 - rr.fun
        bdc = float(rr.x[0])
    else:
        corr, bdc = 0.0, float(event_bdc)

    corrected = (da - bdc * me) / (1 - bdc)
    signal_vals = get_signal(corrected, predicted_density_array > high)
    noise_signal_vals = get_signal(corrected, predicted_mask_array == 1)
    try:
        optimal_signal_contour = get_optimal_signal_contour(signal_vals, noise_signal_vals)
    except Exception:
        optimal_signal_contour = 1.0

    optimized_native = _translate_structure(optimized_local, box_origin)
    num_contacts = get_contacts(optimized_native, structure.structure)
    noise_signal_vals = np.clip(noise_signal_vals, 0.0, 3.0)
    signal_vals = np.clip(signal_vals, 0.0, 3.0)
    save_structure(Structure(None, optimized_native), out_dir / f"{conformer_id}.pdb")
    centroid_native = get_structure_mean(optimized_native)

    return {
        str(out_dir / f"{conformer_id}.pdb"): {
            'score': float(score),
            'centroid': centroid_native,
            'local_signal': float(corr),
            'new_bdc': float(bdc),
            'noise': float(np.abs(np.sum(noise_signal_vals))),
            'signal': float(np.abs(np.sum(signal_vals))),
            'num_points': int(np.sum(predicted_density_array > high)),
            'optimal_contour': float(optimal_signal_contour),
            'num_contacts': int(num_contacts),
            'arr': arr,
        }
    }


def autobuild_conformer(
        centroid,
        event_bdc,
        conformer,
        masked_dtag_array,
        masked_mean_array,
        reference_frame,
        out_dir,
        conformer_id,
        res,
        structure,
        unmasked_dtag_array,
        unmasked_mean_array,
        z_array,
        raw_xmap_sparse,
        score_build,
        raw_xmap_array_ref
):
    # Path X: PANDDA_LOCAL_AUTOBUILD runs the whole build on LOCAL boxes cut from
    # the sparse maps (no full-cell unmask) -> cell-size-independent memory. It is
    # INDEPENDENT of the fit method: inside, score_conformer still picks crowther
    # vs DE via PANDDA_CROWTHER_FIT. So the 2x2 (DE/crowther x full/local) is
    # runnable -- in particular DE+local isolates and validates the memory cut as
    # behaviour-preserving, separate from the crowther fit change.
    if os.environ.get("PANDDA_LOCAL_AUTOBUILD"):
        return _autobuild_conformer_local(
            centroid, event_bdc, conformer, masked_dtag_array, masked_mean_array,
            reference_frame, out_dir, conformer_id, res, structure,
            unmasked_dtag_array, unmasked_mean_array, z_array, raw_xmap_sparse,
            score_build, raw_xmap_array_ref)

    time_begin_autobuild = time.time()

    # event_map_grid = reference_frame.unmask(SparseDMap((masked_dtag_array - (event_bdc*masked_mean_array)) / (1-event_bdc)))
    # print(f'z map stats: {np.min(z_array)} {np.max(z_array)} {np.median(z_array)}')

    event_map_array = (masked_dtag_array - (event_bdc * masked_mean_array)) / (1 - event_bdc)
    normalize_event_map_array = (event_map_array - np.mean(event_map_array) / np.std(event_map_array))
    normalize_z_array = (z_array - np.mean(z_array)) / np.std(z_array)
    normalize_xmap = (masked_dtag_array - np.mean(masked_dtag_array)) / np.std(masked_dtag_array)

    z_grid = reference_frame.unmask(SparseDMap(normalize_z_array))
    # raw_xmap_grid = reference_frame.unmask(SparseDMap(raw_xmap_sparse))
    raw_xmap_grid = gemmi.FloatGrid(*raw_xmap_array_ref.shape)
    raw_xmap_grid.set_unit_cell(z_grid.unit_cell)
    raw_xmap_grid_array = np.array(raw_xmap_grid, copy=False)
    raw_xmap_grid_array[:, :, :] = raw_xmap_array_ref[:, :, :]

    score_grid = np.zeros(normalize_z_array.shape, dtype=np.float32) #normalize_z_array #+ (0.1*masked_dtag_array)
    score_grid[normalize_xmap > 1.5] = 0.5
    score_grid[normalize_z_array > 1.5] = 1.0

    event_map_grid = reference_frame.unmask(SparseDMap(score_grid))

    time_begin_score_conf = time.time()
    optimized_structure, score, centroid, arr = score_conformer(
        centroid,
        conformer.structure,
        event_map_grid,
        score_build,
        z_grid,
        raw_xmap_grid,
        res=res,
    )
    time_finish_score_conf = time.time()

    save_structure(
        Structure(None, optimized_structure),
        out_dir / f"{conformer_id}.pdb",
    )

    xmap = reference_frame.unmask(masked_dtag_array)

    predicted_mask = get_predicted_mask(
        optimized_structure,
        xmap
    )
    predicted_mask_array = np.array(predicted_mask, copy=False)

    predicted_density = get_predicted_density(
        optimized_structure,
        xmap
    )
    predicted_density_array = np.array(predicted_density, copy=False)

    try:
        predicted_density_high_contour = get_predicted_density_high_contour(
            predicted_density,
            predicted_mask
        )
    except:
        predicted_density_high_contour = 1.0

    time_begin_optimize_bdc = time.time()
    corr, bdc = get_local_signal_dencalc_optimize_bdc(
        optimized_structure,
        predicted_density,
        predicted_mask,
        reference_frame,
        unmasked_dtag_array,
        unmasked_mean_array,
        res, event_bdc
    )
    time_finish_optimize_bdc = time.time()

    corrected_event_map_grid = reference_frame.unmask(
        SparseDMap((unmasked_dtag_array - (bdc * unmasked_mean_array)) / (1 - bdc)))
    corrected_event_map_array = np.array(corrected_event_map_grid, copy=False)

    signal_vals = get_signal(
        corrected_event_map_array,
        predicted_density_array > predicted_density_high_contour
    )
    # signal_z = ...

    noise_signal_vals = get_signal(
        corrected_event_map_array,
        predicted_mask_array == 1
    )
    # noise_singal_z = ...

    try:
        optimal_signal_contour = get_optimal_signal_contour(
            signal_vals,
            noise_signal_vals
        )
    except:
        optimal_signal_contour = 1.0

    optimal_noise = np.sum(noise_signal_vals > optimal_signal_contour)
    optimal_signal = np.sum(signal_vals > optimal_signal_contour)

    num_contacts = get_contacts(
        optimized_structure,
        structure.structure
    )

    noise_signal_vals[noise_signal_vals < 0] = 0.0
    noise_signal_vals[noise_signal_vals > 3] = 3.0
    signal_vals[signal_vals < 0] = 0.0
    signal_vals[signal_vals > 3] = 3.0

    time_begin_scoring = time.time()
    # score = score_build(
    #     optimized_structure,
    #     # corrected_event_map_grid,
    #     reference_frame.unmask(SparseDMap(unmasked_dtag_array)),
    #     reference_frame.unmask(SparseDMap(unmasked_mean_array)),
    #     # bdc,
    #     event_bdc,
    #     z_grid,
    #     raw_xmap_grid,
    # )


    x, y, z = centroid
    # print(f'\t\t{round(x, 2)},{round(y, 2)},{round(z, 2)} : {score}')

    time_finish_scoring = time.time()

    log_result_dict = {
        str(out_dir / f"{conformer_id}.pdb"): {
            'score': float(score),
            'centroid': centroid,
            # 'local_signal': get_local_signal(optimized_structure, event_map_grid)
            # 'local_signal': get_local_signal_dencalc(
            #     optimized_structure,
            #     event_map_grid,
            #     res,
            # )
            'local_signal': float(corr),
            'new_bdc': float(bdc),
            # 'new_bdc': float(event_bdc),
            # 'noise': float(noise_signal),
            # 'signal': float(signal),
            # 'noise': float(np.abs(np.median(noise_signal_vals))),
            # 'signal': float(np.abs(np.median(signal_vals))),
            'noise': float(np.abs(np.sum(noise_signal_vals))),
            'signal': float(np.abs(np.sum(signal_vals))),
            'num_points': int(np.sum(predicted_density_array > predicted_density_high_contour)),
            'optimal_contour': float(optimal_signal_contour),
            'num_contacts': int(num_contacts),
            'arr': arr
            # 'total_noise':
        }
    }
    time_finish_autobuild = time.time()
    # print(
    #     f"Autobuilt: {round(time_finish_autobuild - time_begin_autobuild, 2)}. Scored: {round(time_finish_scoring - time_begin_scoring, 2)}. Built: {round(time_finish_score_conf - time_begin_score_conf, 2)}. Optimized BDC: {round((time_finish_optimize_bdc - time_begin_optimize_bdc))}")

    # Return results
    return log_result_dict


class AutobuildModelEventInbuilt:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(
            self,
            event: EventInterface,
            dataset: DatasetInterface,
            dmap,
            mtz,
            model,
            ligand_files,
            out_dir,
    ):

        # Get the structure
        st = Structure.from_path(str(dataset.structure.path))

        # Get the scoring grid
        score_grid = get_score_grid(dmap, st, event)

        # Generate conformers to score
        conformers = get_conformers(ligand_files)
        if len(conformers) == 0:
            return AutobuildResult(
                {},
                None,
                None,
                None,
                ligand_files.ligand_cif,
                out_dir
            )

        # Score conformers against the grid
        conformer_scores = {}
        for conformer_id, conformer in conformers.items():
            optimized_structure, score, centroid = score_conformer(
                np.mean(event.pos_array, axis=0),
                conformer,
                score_grid,
            )
            conformer_scores[conformer_id] = [optimized_structure, score, centroid]

        if len(conformer_scores) == 0:
            return AutobuildResult(
                {},
                None,
                None,
                None,
                ligand_files.ligand_cif,
                out_dir
            )

        # Choose the best ligand
        for conformer_id, (optimized_structure, score, centroid) in conformer_scores.items():
            save_structure(
                Structure(None, optimized_structure),
                out_dir / f"{conformer_id}.pdb",
            )

        log_result_dict = {
            str(out_dir / f"{conformer_id}.pdb"): {'score': score,
                                                   'centroid': centroid, }
            for conformer_id, (optimized_structure, score, centroid)
            in conformer_scores.items()
        }

        # Return results
        return AutobuildResult(
            log_result_dict,
            None,
            None,
            None,
            ligand_files.ligand_cif,
            out_dir
        )
