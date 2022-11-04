from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import numpy as np
import pandas as pd

from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit import rdBase


rdDepictor.SetPreferCoordGen(False)

from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
IPythonConsole.ipython_useSVG = True  # Change output to SVG
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.drawOptions.minFontSize = 18
IPythonConsole.drawOptions.prepareMolsBeforeDrawing = True
rdBase.rdkitVersion

# ---------------------------------------------------------------------------------------------

def read_smiles_file(file):
    smiles_dict = {}
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith(' '):
                continue
            name, smiles = line.rstrip().split()
            smiles_dict[name] = smiles
    return smiles_dict

# ---------------------------------------------------------------------------------------------

def smiles_file_to_df(filename):
    df = pd.read_csv(filename, sep=' ')
    return df


# ---------------------------------------------------------------------------------------------

#https://github.com/CalebBell/thermo/issues/22
def draw_3d(mol, width=300, height=300, style='stick', Hs=True): # pragma: no cover
        '''Interface for drawing an interactive 3D view of the molecule.
        Requires an HTML5 browser, and the libraries RDKit, pymol3D, and
        IPython. An exception is raised if all three of these libraries are
        not installed.
        Parameters
        ----------
        width : int
            Number of pixels wide for the view
        height : int
            Number of pixels tall for the view
        style : str
            One of 'stick', 'line', 'cross', or 'sphere'
        Hs : bool
            Whether or not to show hydrogen
        Examples
        --------
        >>> Chemical('cubane').draw_3d()
        <IPython.core.display.HTML object>
        '''
        try:
            import py3Dmol
            from IPython.display import display
            #AllChem.EmbedMultipleConfs(mol)
            mb = Chem.MolToMolBlock(mol)
            p = py3Dmol.view(width=width,height=height)
            p.addModel(mb,'sdf')
            p.setStyle({style:{}})
            p.addPropertyLabels("index","","")
            p.zoomTo()
            display(p.show())
        except:
            return 'py3Dmol, RDKit, and IPython are required for this feature.'

# ---------------------------------------------------------------------------------------------

def draw_mols_from_smiles(smiles_list):
    display(Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in smiles_list]))
    for smi in smiles_list:
        m1 = Chem.MolFromSmiles(smi)
        Chem.AssignStereochemistry(m1)
        display(m1)
        print("mapped smiles = ", smi)
        print("canonical_smiles = ", remove_label_chirality(smi))
        m1 = Chem.AddHs(m1)
        AllChem.EmbedMolecule(m1, randomSeed=0)
        draw_3d(m1)
    return

# ---------------------------------------------------------------------------------------------
# MAPPING
# ---------------------------------------------------------------------------------------------

def remove_label_chirality(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    
    # Remove atom mapping
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    
    # Reassign stereochemistry 
    rdmolops.AssignStereochemistry(mol, cleanIt=True, flagPossibleStereoCenters=True, force=True)
    
    return Chem.MolToSmiles(mol)

# ---------------------------------------------------------------------------------------------

def remove_label_chirality_v2(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    
    # Remove atom mapping
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    
    # Reassign stereochemistry 
    rdmolops.AssignStereochemistry(mol, cleanIt=True, flagPossibleStereoCenters=True, force=True)
    
    return Chem.MolToSmiles(mol)

# ---------------------------------------------------------------------------------------------

def reorder_atoms_to_map(mol):

    """
    Reorders the atoms in a mol objective to match that of the mapping
    """

    atom_map_order = np.zeros(mol.GetNumAtoms()).astype(int)
    for atom in mol.GetAtoms():
        map_number = atom.GetAtomMapNum()-1
        atom_map_order[map_number] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, atom_map_order.tolist())
    return mol

# ---------------------------------------------------------------------------------------------

def get_mapped_smiles(mol):
    """
    label all potential stereocenters and choose one stereosiomer
    """
    for i, atom in enumerate(mol.GetAtoms()):
        a = mol.GetAtomWithIdx(i)
        a.SetAtomMapNum(i+1)
    rdmolops.AssignStereochemistry(mol, cleanIt=True, flagPossibleStereoCenters=True, force=True)

    mol_isomer = next(EnumerateStereoisomers(mol))
    rdmolops.AssignStereochemistry(mol_isomer, force=True)
    mol_isomer_smiles = Chem.MolToSmiles(mol_isomer)

    return mol_isomer, mol_isomer_smiles

# ---------------------------------------------------------------------------------------------

rm_proton = ( ('[CX4;H3]','[CH2-]'),('[CX4;H2]','[CH-]'),('[CX4;H1]','[C-]'),('[CX3;H2]','[CH-]'),
              ('[CX3;H1]','[C-]'), ('[CX2;H1]','[C-]'))

rm_hydride =( ('[CX4;H3]','[CH2+]'),('[CX4;H2]','[CH+]'),('[CX4;H1]','[C+]'),('[CX3;H2]','[CH+]'),
              ('[CX3;H1]','[C+]'), ('[CX2;H1]','[C+]'))

rm_hydrogen = ( ('[CX4;H3]','[CH2]'),('[CX4;H2]','[CH]'),('[CX4;H1]','[C]'),('[CX3;H2]','[CH]'),
              ('[CX3;H1]','[C]'), ('[CX2;H1]','[C]'))

add_proton_hydrogen = ['[C;H1:1]=[C,N;H1:2]>>[CH2:1][*H+:2]', '[C;H1:1]=[C,N;H0:2]>>[CH2:1][*+;H0:2]']

# ---------------------------------------------------------------------------------------------

def define_conditions(rxn=None):

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
          
     return smartsref, delta_charge


# ---------------------------------------------------------------------------------------------
#TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------


def check_chiral(m,ion):
    chiral_atoms_mol = Chem.FindMolChiralCenters(m)
    
    number_of_chiral_atoms_mol = len(chiral_atoms_mol)
    if number_of_chiral_atoms_mol == 0:
        return ion

    #ion = Chem.MolFromSmiles(ion_smiles)
    #Chem.FindMolChiralCenters(ion, useLegacyImplementation=False, includeUnassigned=True)
    chiral_atoms_ion = Chem.FindMolChiralCenters(ion)

    for chiral_atom_mol,chiral_atom_ion in zip(chiral_atoms_mol,chiral_atoms_ion):
        if chiral_atom_mol != chiral_atom_ion:
            ion.GetAtomWithIdx(chiral_atom_ion[0]).InvertChirality()


    return ion

# ---------------------------------------------------------------------------------------------

def transform_mol_v4(smartsref, delta_charge, ref_mol, show_mol=False):
    lst_atoms = []
    lst_new_mols = []
    new_smiles = []
    lst_atommapnum = []

    ref_mol_kekulized = copy.deepcopy(ref_mol)
    Chem.Kekulize(ref_mol_kekulized,clearAromaticFlags=True)

    for (smarts, smiles) in smartsref:
        patt1 = Chem.MolFromSmarts(smarts)
        patt2 = Chem.MolFromSmiles(smiles)
        atoms = ref_mol_kekulized.GetSubstructMatches(patt1)
      
        if len(atoms) != 0:
            [lst_atoms.append(element) for tupl in atoms for element in tupl]
            new_mol = Chem.rdmolops.ReplaceSubstructs(ref_mol_kekulized, patt1, patt2)
            lst_new_mols += new_mol

    lst_new_mol_chiral = [check_chiral(ref_mol_kekulized,s) for s in lst_new_mols]
    
    for a_index, mol in zip(lst_atoms, lst_new_mol_chiral):
        a = mol.GetAtomWithIdx(mol.GetNumAtoms()-1)
        a.SetAtomMapNum(a_index+1)
        lst_atommapnum.append(a.GetAtomMapNum())
        s = Chem.MolToSmiles(mol,isomericSmiles=True, canonical=False, kekuleSmiles=True)
        new_smiles.append(s)

    # for idx, smi in enumerate(new_smiles):
    #     m1 = Chem.MolFromSmiles(smi)
    #     Chem.AssignStereochemistry(m1)
    #     lst_atommapnum += [atom.GetAtomMapNum() for atom in m1.GetAtoms() if atom.GetFormalCharge() != 0]

    return new_smiles, lst_atommapnum

# ---------------------------------------------------------------------------------------------


def find_and_reduce_smiles_v3(smiles_list=None, details=False):
    lst_reduced_atommapnum = []
    lst_canon_smiles = [remove_label_chirality_v2(smi) for smi in smiles_list]
    unique_smis, uniqueIdx = np.unique(lst_canon_smiles, return_index=True)
    lst_reduced_smiles = [smiles_list[idx] for idx in uniqueIdx.tolist()]

    for smi in lst_reduced_smiles:
        m1 = Chem.MolFromSmiles(smi)
        Chem.AssignStereochemistry(m1)
        lst_reduced_atommapnum += [atom.GetAtomMapNum() for atom in m1.GetAtoms() if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0]

    if details:
        print('--'*80)
        print(f'canonical smiles: {lst_canon_smiles}')
        print('Reduced list of smiles')
        print(lst_reduced_smiles)
        print('--'*80)
        display(Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in lst_reduced_smiles], subImgSize=(200,200)))
        print('--'*80)
        draw_mols_from_smiles(lst_reduced_smiles)
    
    return lst_reduced_smiles, lst_reduced_atommapnum

# ---------------------------------------------------------------------------------------------

def change_mol_v6(name, smiles, rxn, reduce_smiles=False, show_mol=False):
    ref_mol = Chem.MolFromSmiles(smiles)
    ref_mol_isomer, ref_mol_isomer_smiles = get_mapped_smiles(ref_mol)
    ref_mol_isomer_kekulized = copy.deepcopy(ref_mol_isomer)
    Chem.Kekulize(ref_mol_isomer_kekulized,clearAromaticFlags=True)
    charge = Chem.GetFormalCharge(ref_mol_isomer_kekulized)


    smartsref, delta_charge = define_conditions(rxn=rxn)

    if delta_charge != 0:
        new_charge = "#" + str(charge+delta_charge)
    else:
        new_charge = "-rad#" + str(charge)

    lst_new_smiles, lst_atommapnum = transform_mol_v4(smartsref, delta_charge, ref_mol_isomer, show_mol=False)
    count_new_smiles = len(lst_new_smiles)

    if reduce_smiles:
        lst_new_smiles, lst_atommapnum = find_and_reduce_smiles_v3(lst_new_smiles, details=False)
        count_new_smiles_reduced = len(lst_new_smiles)

    lst_new_smiles_nomap = [remove_label_chirality(smi) for smi in lst_new_smiles]
    lst_canon_smiles = [remove_label_chirality_v2(smi) for smi in lst_new_smiles]

    legend = list(zip(lst_canon_smiles, lst_atommapnum))
        
    if show_mol:
        print('--'*80)
        print(f'{name} reference molecule')
        print(ref_mol_isomer_smiles)
        display(ref_mol_isomer)
        display(Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in lst_new_smiles], legends=[f'{a} {str(b)}' for a,b in legend], subImgSize=(200,200)))
        # draw_mols_from_smiles(lst_new_smiles)

    # print(f'atom list: {lst_atommapnum}')
    # print(f'Generated {count_new_smiles} new smiles')
    # if reduce_smiles:
    #     print(f'After reduction, {count_new_smiles_reduced} new smiles')

    lst_new_name = [f"{name}{new_charge}={str(a)}" for a in lst_atommapnum]

    return lst_new_name, lst_new_smiles, lst_new_smiles_nomap, lst_canon_smiles, lst_atommapnum

# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    import time
    start = time.perf_counter()

    input_smiles = 'O1C=CC=C1'
    name = 'furan' 
    # input_mol = Chem.MolFromSmiles('c1c(c2cc(sc2)C)n[nH]c1')
    # input_mol = Chem.MolFromSmiles('c1cc(oc1)CCCN1C(=O)C[C@@H](C1=O)O')
    # input_mol = Chem.MolFromSmiles('c1ccccc1O')

    print(change_mol_v6(name=name, smiles=input_smiles, rxn='rm_proton', reduce_smiles=True, show_mol=False))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')