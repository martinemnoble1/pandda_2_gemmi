from pathlib import Path

import gemmi
from rdkit import Chem

# Legacy text-keyed bond map. The cif reader below now relies on gemmi's typed
# bond parser (GEMMI_BOND_TYPE_TO_RDKIT) instead, but this is retained for any
# external callers that still import it.
bond_type_cif_to_rdkit = {
    'single': Chem.rdchem.BondType.SINGLE,
    'double': Chem.rdchem.BondType.DOUBLE,
    'triple': Chem.rdchem.BondType.TRIPLE,
    'SINGLE': Chem.rdchem.BondType.SINGLE,
    'DOUBLE': Chem.rdchem.BondType.DOUBLE,
    'TRIPLE': Chem.rdchem.BondType.TRIPLE,
    'aromatic': Chem.rdchem.BondType.AROMATIC,
    # 'deloc': Chem.rdchem.BondType.OTHER
    'deloc': Chem.rdchem.BondType.SINGLE,
}

# gemmi.ChemComp reports bond orders as a typed enum, normalised across the
# monomer-dictionary dialects (acedrg/grade/refmac/PDBx) -- so we no longer have
# to guess which spelling of the bond-order column a given dict used.
GEMMI_BOND_TYPE_TO_RDKIT = {
    gemmi.BondType.Single: Chem.rdchem.BondType.SINGLE,
    gemmi.BondType.Double: Chem.rdchem.BondType.DOUBLE,
    gemmi.BondType.Triple: Chem.rdchem.BondType.TRIPLE,
    gemmi.BondType.Aromatic: Chem.rdchem.BondType.AROMATIC,
    # Treat delocalised/metal bonds as single (matches the legacy behaviour).
    gemmi.BondType.Deloc: Chem.rdchem.BondType.SINGLE,
    gemmi.BondType.Metal: Chem.rdchem.BondType.SINGLE,
}


def get_comp_block_key(cif):
    """Resolve the name of the ligand restraint block in a _chem_comp cif.

    A monomer dictionary contains a "comp_list" header block plus the actual
    restraint block named "comp_<TLC>" (e.g. comp_LIG, comp_DRG). The
    three-letter code varies between dictionary generators (acedrg, grade,
    refmac, ...), so read the block from the document rather than guessing the
    TLC: return the first block (other than the comp_list header) that contains
    a _chem_comp_atom loop. Falls back to the historical comp_LIG/comp_XXX
    guesses for unusual layouts.
    """
    for block in cif:
        if block.name == "comp_list":
            continue
        if list(block.find_loop('_chem_comp_atom.atom_id')):
            return block.name

    for candidate in ("comp_LIG", "comp_XXX"):
        try:
            cif[candidate]
            return candidate
        except Exception:
            continue

    raise KeyError(
        "No _chem_comp restraint block found in cif "
        f"(blocks present: {[block.name for block in cif]})"
    )


def get_fragment_mol_from_dataset_cif_path(dataset_cif_path: Path):
    """Build an RDKit mol (connectivity only) from a ligand monomer dictionary.

    Atoms and bonds are read via gemmi's ``ChemComp`` parser. gemmi understands
    every monomer-dictionary dialect (acedrg/grade/refmac/PDBx) and, crucially,
    *both* spellings of the bond-order column: modern acedrg/PDBx dicts write
    ``_chem_comp_bond.value_order`` (+ ``pdbx_aromatic_flag``) while older
    refmac/grade dicts write ``_chem_comp_bond.type`` (+ ``aromatic``).

    The previous hand-rolled reader only looked for ``_chem_comp_bond.type``.
    On an acedrg PDBx dict that loop came back empty, so the ``zip()`` over the
    bond columns yielded nothing and the molecule was built with **no bonds** --
    a bag of disconnected atoms. RDKit then "embedded" every atom at the origin
    and the downstream autobuild dropped the whole collapsed blob onto the event
    centroid, so the modelled ligand appeared as a single point. Letting gemmi
    read the block fixes this for every dialect at once (and makes the old
    sulfonate bond-order fix-up unnecessary, since gemmi reports the declared
    S=O double bonds directly).

    The returned mol carries no conformer; callers generate 3D coordinates with
    RDKit (see ``autobuild.inbuilt.get_conformers``).
    """
    cif = gemmi.cif.read(str(dataset_cif_path))
    key = get_comp_block_key(cif)
    chem_comp = gemmi.make_chemcomp_from_block(cif[key])

    editable_mol = Chem.EditableMol(Chem.Mol())

    id_to_idx = {}
    for idx, atom in enumerate(chem_comp.atoms):
        rd_atom = Chem.Atom(atom.el.name)
        rd_atom.SetFormalCharge(round(atom.charge))
        editable_mol.AddAtom(rd_atom)
        id_to_idx[atom.id] = idx

    for bond in chem_comp.rt.bonds:
        if bond.aromatic:
            order = Chem.rdchem.BondType.AROMATIC
        else:
            order = GEMMI_BOND_TYPE_TO_RDKIT.get(
                bond.type, Chem.rdchem.BondType.SINGLE)
        editable_mol.AddBond(
            id_to_idx[bond.id1.atom],
            id_to_idx[bond.id2.atom],
            order=order,
        )

    mol = editable_mol.GetMol()

    # Prefer a full sanitize -- it perceives the rings/aromaticity/valences that
    # RDKit conformer embedding relies on. Fall back to the lenient
    # property-cache update used historically so we never regress an unusual
    # ligand that a strict sanitize would reject.
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.MolSanitizeException:
        mol.UpdatePropertyCache(strict=False)

    return mol
