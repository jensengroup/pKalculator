import copy
import numpy as np
import re
import random
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


# ---------------------------------------------------------------------------------------------
# MAPPING
# ---------------------------------------------------------------------------------------------


def remove_atom_mapping(smiles: str) -> str:
    """
    Remove the atom mapping of a reaction SMILES.
    The resulting SMILES strings will still contain brackets and it may be
    advisable to canonicalize them or clean them up as a postprocessing step.
    Args:
        smiles: SMILES string potentially containing mapping information.
    Returns:
        A SMILES string without atom mapping information.
    """

    # We look for ":" followed by digits before a "]" not coming after an "*"
    return re.sub(r"(?<=[^\*])(:\d+)]", "]", smiles)


def reorder_atoms_to_map(mol):
    """
    Reorders the atoms in a mol objective to match that of the mapping
    """

    atom_map_order = np.zeros(mol.GetNumAtoms()).astype(int)
    for atom in mol.GetAtoms():
        map_number = atom.GetAtomMapNum() - 1
        atom_map_order[map_number] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, atom_map_order.tolist())
    return mol


def remove_mapping(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi, sanitize=True)

    # Remove atom mapping
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]

    # Reassign stereochemistry
    # rdmolops.AssignStereochemistry(mol, cleanIt=True, flagPossibleStereoCenters=True, force=True)
    rdmolops.FindPotentialStereo(mol, cleanIt=True, flagPossible=True)

    return Chem.MolToSmiles(mol)


def get_mapped_smiles(mol):
    """
    label all potential stereocenters and choose one stereosiomer
    to ensure all deprotonations are done on the same stereoisomer
    """
    for i, _ in enumerate(mol.GetAtoms()):
        a = mol.GetAtomWithIdx(i)
        a.SetAtomMapNum(i + 1)
    rdmolops.AssignStereochemistry(
        mol, cleanIt=True, flagPossibleStereoCenters=True, force=True
    )

    mol_isomer = next(
        EnumerateStereoisomers(
            mol, StereoEnumerationOptions(tryEmbedding=True, unique=True)
        )
    )
    rdmolops.AssignStereochemistry(mol_isomer, force=True)
    mol_isomer_smiles = Chem.MolToSmiles(mol_isomer)

    return mol_isomer, mol_isomer_smiles


# ---------------------------------------------------------------------------------------------


def define_conditions(rxn=None):
    rm_proton = (
        ("[CX4;H4]", "[CH3-]"),
        ("[CX4;H3]", "[CH2-]"),
        ("[CX4;H2]", "[CH-]"),
        ("[CX4;H1]", "[C-]"),
        ("[CX3;H2]", "[CH-]"),
        ("[CX3;H1]", "[C-]"),
        ("[CX2;H1]", "[C-]"),
    )

    rm_hydride = (
        ("[CX4;H4]", "[CH3-]"),
        ("[CX4;H3]", "[CH2+]"),
        ("[CX4;H2]", "[CH+]"),
        ("[CX4;H1]", "[C+]"),
        ("[CX3;H2]", "[CH+]"),
        ("[CX3;H1]", "[C+]"),
        ("[CX2;H1]", "[C+]"),
    )

    rm_hydrogen = (
        ("[CX4;H4]", "[CH3-]"),
        ("[CX4;H3]", "[CH2]"),
        ("[CX4;H2]", "[CH]"),
        ("[CX4;H1]", "[C]"),
        ("[CX3;H2]", "[CH]"),
        ("[CX3;H1]", "[C]"),
        ("[CX2;H1]", "[C]"),
    )

    add_proton_hydrogen = (
        "[C;H1:1]=[C,N;H1:2]>>[CH2:1][*H+:2]",
        "[C;H1:1]=[C,N;H0:2]>>[CH2:1][*+;H0:2]",
    )

    rm_NO_proton = (
        ("[NX3;H1;!$([NX3+])]", "[N-]"),
        ("[NX3;H2]", "[NH-]"),
        ("[NX3;H3]", "[NH2-]"),
        ("[NX4+;H1]", "[N]"),
        ("[NX4+;H2]", "[NH]"),
        ("[NX4+;H3]", "[NH2]"),
        ("[NX4+;H4]", "[NH3]"),
        ("[NX3+;H1]", "[N]"),
        (
            "[$([nX3](*[H]):*),$([nX2](*[H]):*),$([#7X2;H1]=*),$([NX3](*[H])(:*)),$([#7X3+](-[*]):*),$([#7X3+H](:*))]",
            "[N-]",
        ),
        ("[$([N+]#*)]", "[N]"),
        ("[OX2;H1]", "[O-]"),
        ("[OX3+;H1]", "[O]"),
        ("[OX3+;H2]", "[OH]"),
        ("[SX2;H1]", "[S-]"),
        ("[SX4;H1]", "[S-]"),
        ("[SX6;H1]", "[S-]"),
        ("[S+;H2]", "[SH]"),
        ("[S+;H1]", "[S]"),
        ("[SeX2;H1]", "[Se-]"),
        ("[GeX4;H1]", "[Ge-]"),
    )

    rm_all_protons = rm_proton + rm_NO_proton

    if rxn == "rm_proton":
        smartsref = rm_proton
        delta_charge = -1
    if rxn == "rm_hydride":
        smartsref = rm_hydride
        delta_charge = 1
    if rxn == "rm_hydrogen":
        smartsref = rm_hydrogen
        delta_charge = 0
    if rxn == "add_proton":
        smartsref = add_proton_hydrogen
        delta_charge = 1
    if rxn == "add_hydrogen":
        smartsref = add_proton_hydrogen
        delta_charge = 0
    if rxn == "add_electron":
        delta_charge = -1
    if rxn == "rm_electron":
        delta_charge = +1

    if rxn == "rm_NO_proton":
        smartsref = rm_NO_proton
        delta_charge = -1
    if rxn == "rm_all_protons":
        smartsref = rm_all_protons
        delta_charge = -1

    return smartsref, delta_charge


# ---------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------


def check_same_chirality(smiles_mol: str, smiles_ion: str) -> str:
    """Replace differences in chirality between two SMILES strings.

    Args:
        smiles_mol (str): The SMILES string of the molecule.
        smiles_ion (str): The SMILES string of the ion.

    Returns:
        str: The modified SMILES string of the ion.
    """
    # Define the regular expression pattern
    pattern = r"([A-Za-z0-9])(\@+)([A-Za-z0-9]:|:)(\d+)"

    # Find all matches in the SMILES strings
    lst_deprot_chiral = re.findall(pattern, smiles_ion)
    lst_chiral = re.findall(pattern, smiles_mol)
    # Create a dictionary of chiral matches in the molecule SMILES string
    chiral_dict = {match[3]: match for match in lst_chiral}

    # Replace differences in chirality in the ion SMILES string
    for match_ion in lst_deprot_chiral:
        match_mol = chiral_dict.get(match_ion[3])
        if match_mol and match_ion != match_mol:
            str_match_ion = "".join(match_ion)
            str_match_mol = "".join(match_mol)
            pattern_to_replace = re.compile(f"\[{str_match_ion}\]")
            # smi_ion = re.sub(pattern, '[C@@H:3]', smi_ion)
            # smiles_ion = smiles_ion.replace("".join(match_ion), "".join(match_mol))
            smiles_ion = re.sub(pattern_to_replace, f"[{str_match_mol}]", smiles_ion)

    return smiles_ion


def deprotonate(name: str, smiles: str, rxn: str):
    """Generates a list of smiles by replacing an atom based on the smartsref pattern.
    Args:
        name (str): name of the molecule
        smiles (str): smiles string of the molecule
        rxn (str): e.g 'rm_proton' to deprotonate a molecule

    Returns:
            ref_mol_isomer_smiles (str): the smiles string of the isomer of the reference molecule
            lst_new_name (list): list of names of the deprotonated molecules
            lst_new_smiles2 (list): list of smiles strings of the deprotonated molecules
            lst_new_smiles_nomap (list): list of smiles strings of the deprotonated molecules without atom mapping
            lst_atommapnum (list): list of atom map numbers of the deprotonated molecules

    example:
    deprotonate('test', 'F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2', "rm_proton")

    ('[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
    ['test#-1=3', 'test#-1=6', 'test#-1=7', 'test#-1=8', 'test#-1=9'],
    ['[F:1][C@@:2]12[CH-:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
    '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH-:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
    '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C-:7]([CH:8]=[CH:9][Br:10])[O:11]2',
    '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C@@H:7]([C-:8]=[CH:9][Br:10])[O:11]2',
    '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C@@H:7]([CH:8]=[C-:9][Br:10])[O:11]2'],
    ['F[C@@]12[CH-][C@]1(Cl)C[C@@H](/C=C/Br)O2',
    'F[C@@]12C[C@]1(Cl)[CH-][C@@H](/C=C/Br)O2',
    'F[C@@]12C[C@]1(Cl)C[C-](C=CBr)O2',
    'F[C@@]12C[C@]1(Cl)C[C@@H]([C-]=CBr)O2',
    'F[C@@]12C[C@]1(Cl)C[C@@H](C=[C-]Br)O2'],
    [3, 6, 7, 8, 9])
    """
    lst_atoms = []
    lst_new_mols = []
    lst_new_smiles = []
    lst_atommapnum = []
    lst_atomidx = []
    lst_new_mols_corrected = []

    ref_mol = Chem.MolFromSmiles(smiles)
    # generate a random isomer and use this as the reference molecule
    ref_mol_isomer, ref_mol_isomer_smiles = get_mapped_smiles(ref_mol)

    ref_mol_isomer_kekulized = copy.deepcopy(ref_mol_isomer)
    Chem.Kekulize(ref_mol_isomer_kekulized, clearAromaticFlags=True)
    charge = Chem.GetFormalCharge(ref_mol_isomer_kekulized)

    smartsref, delta_charge = define_conditions(rxn=rxn)

    if delta_charge != 0:
        new_charge = "#" + str(charge + delta_charge)
    else:
        new_charge = "-rad#" + str(charge)

    # display(ref_mol_isomer_kekulized)
    # print(f'ref_mol_isomer_smiles: {ref_mol_isomer_smiles}')

    for smarts, smiles in smartsref:
        patt1 = Chem.MolFromSmarts(smarts)
        patt2 = Chem.MolFromSmiles(smiles)
        atoms = ref_mol_isomer_kekulized.GetSubstructMatches(patt1)
        # print(f'smarts: {smarts}, smiles: {smiles}, atoms: {atoms}')

        try:
            if len(atoms) != 0:
                # [lst_atoms.append(element) for tupl in atoms for element in tupl]
                # Flatten list of tuples and add atoms to lst_atoms
                lst_atoms += [element for tupl in atoms for element in tupl]
                new_mol = Chem.rdmolops.ReplaceSubstructs(
                    ref_mol_isomer_kekulized, patt1, patt2
                )
                lst_new_mols += new_mol
        except Exception as e:
            print(f"Error for replacing for {smarts}: {e}")

    # Check for unique smiles
    lst_canon_smiles = [remove_mapping(Chem.MolToSmiles(mol)) for mol in lst_new_mols]

    # find the unique smiles and sort them based on the deprotonated atom number
    _, uniqueIdx = np.unique(lst_canon_smiles, return_index=True)
    lst_reduced_atoms_mols = sorted(
        [(lst_atoms[idx], lst_new_mols[idx]) for idx in uniqueIdx], key=lambda x: x[0]
    )

    for a_index, mol in lst_reduced_atoms_mols:
        a = mol.GetAtomWithIdx(mol.GetNumAtoms() - 1)
        # a = mol.GetAtomWithIdx(a_index)
        a.SetAtomMapNum(a_index + 1)
        lst_atommapnum.append(a.GetAtomMapNum())
        lst_atomidx.append(a_index)
        # s = Chem.MolToSmiles(mol,isomericSmiles=True, canonical=False, kekuleSmiles=True)
        lst_new_smiles.append(Chem.MolToSmiles(mol))
        lst_new_mols_corrected.append(mol)
    # Checks if the chirality is the same as the initial molecule
    lst_new_smiles_2 = [
        check_same_chirality(ref_mol_isomer_smiles, deprot_smiles)
        for deprot_smiles in lst_new_smiles
    ]

    # in the future one may want to use remove_atom_mapping(smiles) to remove atom mapping
    lst_new_smiles_nomap = [remove_mapping(smi) for smi in lst_new_smiles_2]
    lst_new_name = [f"{name}{new_charge}={str(a)}" for a in lst_atommapnum]

    return (
        ref_mol_isomer_smiles,
        lst_new_name,
        lst_new_smiles_2,
        lst_new_smiles_nomap,
        lst_atommapnum,
        lst_atomidx,
    )


def remove_Hs(
    name=None, smiles=None, rdkit_mol=None, atomsite=None, gen_all=False, remove_H=False
):
    """_summary_
    Deprotonates a molecule at a specific atom site and removes the hydrogens if atomsite and gen_all=True is invoked.
    Example:

    remove_Hs(name='test', smiles='F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2', atomsite=3, gen_all=False, remove_H=True)
    Output:
        ((2,),
        (3,),
        ('F[C@@]12[CH-][C@]1(Cl)C[C@@H](/C=C/Br)O2',),
        ('[F:1][C@@:2]12[CH-:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',),
        (<rdkit.Chem.rdchem.Mol at 0x7f156c838dd0>,),
        ('test#-1=3',),
        {2: [11, 12]})

    if gen_all == True, then all posible deprotonations on the C atom will be generated.
    if remove_H == True, then the hydrogens will be removed from the molecule.

    Example:
    remove_Hs(name='test', smiles='F[C@@]12C[C@]1(Cl)C[C@@H](/C=C/Br)O2', atomsite=None, gen_all=True, remove_H=True)
    Output:
        ((2, 5, 6, 7, 8),
        (3, 6, 7, 8, 9),
        ('F[C@@]12[CH-][C@]1(Cl)C[C@@H](/C=C/Br)O2',
        'F[C@@]12C[C@]1(Cl)[CH-][C@@H](/C=C/Br)O2',
        'F[C@@]12C[C@]1(Cl)C[C-](/C=C/Br)O2',
        'F[C@@]12C[C@]1(Cl)C[C@@H](/[C-]=C/Br)O2',
        'F[C@@]12C[C@]1(Cl)C[C@@H](/C=[C-]/Br)O2'),
        ('[F:1][C@@:2]12[CH-:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
        '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH-:6][C@@H:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
        '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C-:7](/[CH:8]=[CH:9]/[Br:10])[O:11]2',
        '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[C-:8]=[CH:9]/[Br:10])[O:11]2',
        '[F:1][C@@:2]12[CH2:3][C@:4]1([Cl:5])[CH2:6][C@@H:7](/[CH:8]=[C-:9]/[Br:10])[O:11]2'),
        (<rdkit.Chem.rdchem.Mol at 0x7f156c83a0a0>,
        <rdkit.Chem.rdchem.Mol at 0x7f156c839930>,
        <rdkit.Chem.rdchem.Mol at 0x7f156c838970>,
        <rdkit.Chem.rdchem.Mol at 0x7f156c83aab0>,
        <rdkit.Chem.rdchem.Mol at 0x7f156c83a7a0>),
        ('test#-1=3', 'test#-1=6', 'test#-1=7', 'test#-1=8', 'test#-1=9'),
        {2: [11, 12], 5: [13, 14], 6: [15], 7: [16], 8: [17]})

    Args:
        name (_type_, optional): _description_. Defaults to None.
        smiles (_type_, optional): _description_. Defaults to None.
        rdkit_mol (_type_, optional): _description_. Defaults to None.
        atomsite (_type_, optional): _description_. Defaults to None.
        gen_all (bool, optional): _description_. Defaults to False.
        remove_H (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    random.seed(42)
    if smiles:
        rdkit_mol = Chem.MolFromSmiles(smiles)

    rdkit_mol = Chem.AddHs(rdkit_mol)
    charge_neutral = Chem.GetFormalCharge(rdkit_mol)

    dict_atom_idx_to_H_indices = {
        atom_idx: [
            atom_neighbor.GetIdx()
            for atom_neighbor in atom.GetNeighbors()
            if atom_neighbor.GetSymbol() == "H"
        ]
        for atom_idx, atom in enumerate(rdkit_mol.GetAtoms())
        if atom.GetSymbol() != "H" and atom.GetSymbol() == "C"
    }  # note if atom.GetSymbol() == "C" is note invoked, O-H, N-H, etc. will also get deprotonated.
    dict_atom_idx_to_H_indices = {
        k: v for k, v in dict_atom_idx_to_H_indices.items() if v
    }  # removes empty lists
    lst_idx_deprot_mol = []

    if not gen_all:
        dict_atom_idx_to_H_indices = {
            k: v for k, v in dict_atom_idx_to_H_indices.items() if k == atomsite - 1
        }

    # print('dict')
    # print(dict_atom_idx_to_H_indices)

    for atom_idx, H_indices in dict_atom_idx_to_H_indices.items():
        H_idx = random.choice(H_indices)
        rdkit_mol2 = Chem.RWMol(rdkit_mol)
        rdkit_mol2.RemoveAtom(H_idx)
        rdkit_mol2.GetAtomWithIdx(atom_idx).SetFormalCharge(-1)
        rdkit_mol2 = rdkit_mol2.GetMol()
        if remove_H:
            rdkit_mol2 = Chem.RemoveHs(rdkit_mol2)
        smi2 = Chem.MolToSmiles(rdkit_mol2)
        for i, a in enumerate(rdkit_mol2.GetAtoms()):
            a.SetAtomMapNum(i + 1)
        smi2_map = Chem.MolToSmiles(rdkit_mol2)
        charge_deprot = Chem.GetFormalCharge(rdkit_mol2)
        name_deprot = f"{name}#{charge_neutral+charge_deprot}={atom_idx+1}"
        lst_idx_deprot_mol.append(
            (atom_idx, atom_idx + 1, smi2, smi2_map, rdkit_mol2, name_deprot)
        )

    # Check for unique smiles
    lst_canon_smiles = [
        remove_mapping(Chem.MolToSmiles(mol))
        for mol in [x[4] for x in lst_idx_deprot_mol]
    ]

    # find the unique smiles and sort them based on the deprotonated atom number
    _, uniqueIdx = np.unique(lst_canon_smiles, return_index=True)

    lst_idx_deprot_mol = [
        x for idx, x in enumerate(lst_idx_deprot_mol) if idx in uniqueIdx
    ]
    lst_idx_deprot_mol = sorted(lst_idx_deprot_mol, key=lambda x: x[0])
    (
        lst_atom_index_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
    ) = zip(*lst_idx_deprot_mol)

    return (
        lst_atom_index_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    )


# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    start = time.perf_counter()
    lst_input_smiles = [
        "O1C=CC=C1",
        "c1c(c2cc(sc2)C)n[nH]c1",
        "c1cc(oc1)CCCN1C(=O)C[C@@H](C1=O)O",
        "CC(C)=O",
    ]

    lst_deprotonate = [
        deprotonate(name=f"test{i}", smiles=smi, rxn="rm_proton")
        for i, smi in enumerate(lst_input_smiles)
    ]

    # removeHs
    lst_removeHs = [
        remove_Hs(
            name=f"test{i}",
            smiles=smi,
            rdkit_mol=None,
            atomsite=None,
            gen_all=True,
            remove_H=True,
        )
        for i, smi in enumerate(lst_input_smiles)
    ]

    print("######################")
    for test_deprotonate, test_removeHs in zip(lst_deprotonate, lst_removeHs):
        print("deprotonate")
        print(test_deprotonate)
        print("--" * 20)
        print("removeHs")
        print(test_removeHs)
        print("######################")
        print("\n")

    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
