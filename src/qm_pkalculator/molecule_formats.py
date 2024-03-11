# MIT License
#
# Copyright (c) 2022 Nicolai Ree
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import subprocess

from rdkit import Chem
from rdkit.Chem import rdmolfiles, AllChem
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def shell(cmd, shell=False):

    if shell:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        cmd = cmd.split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, err = p.communicate()
    return output


def convert_xyz_to_sdf(xyzfile, sdffile):

    shell(f'obabel -ixyz {xyzfile} -osdf -xf > {sdffile}', shell=True)

    return


def convert_sdf_to_xyz(sdffile, xyzfile):
    
    shell(f'obabel -isdf {sdffile} -oxyz -xf > {xyzfile}', shell=True)

    return


def get_bonds(sdf_file):
    """ The count line is; aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv,
    where aaa is atom count and bbb is bond count.
    """

    atoms = 0
    bond_list = []

    searchlines = open(sdf_file, 'r').readlines()

    for i, line in enumerate(searchlines):
        words = line.split() #split line into words
        if len(words) < 1:
            continue
        if i == 3:
            atoms = int(line[0:3])
            bonds = int(line[3:6])
        if 'Pd' in words: #find atom index of Pd
            transistion_metal_idx = i - 3
        else:
            transistion_metal_idx = -1
        if i > atoms+3 and i <= atoms+bonds+3:
            atom_1 = int(line[0:3])
            atom_2 = int(line[3:6])
            if (atom_1 == transistion_metal_idx) or (atom_2 == transistion_metal_idx): #skip bonds to Pd
                continue
            if atom_2 > atom_1:
                bond_list.append(tuple((atom_1,atom_2)))
            else:
                bond_list.append(tuple((atom_2,atom_1)))

    bond_list.sort()

    return bond_list


def get_bonds_molblock(molblock):
    """ The count line is; aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv,
    where aaa is atom count and bbb is bond count.
    """

    atoms = 0
    bond_list = []

    searchlines = molblock.split('\n')

    for i, line in enumerate(searchlines):
        words = line.split() #split line into words
        if len(words) < 1:
            continue
        if i == 3:
            atoms = int(line[0:3])
            bonds = int(line[3:6])
        if 'Pd' in words: #find atom index of Pd
            transistion_metal_idx = i - 3
        else:
            transistion_metal_idx = -1
        if i > atoms+3 and i <= atoms+bonds+3:
            atom_1 = int(line[0:3])
            atom_2 = int(line[3:6])
            if (atom_1 == transistion_metal_idx) or (atom_2 == transistion_metal_idx): #skip bonds to Pd
                continue
            if atom_2 > atom_1:
                bond_list.append(tuple((atom_1,atom_2)))
            else:
                bond_list.append(tuple((atom_2,atom_1)))

    bond_list.sort()

    return bond_list


def compare_sdf_structure(start, end, molblockStart=False):
    """
    Returns True if structures are the same

    Return False if there has been a proton transfer
    """
    
    if molblockStart:
        bond_start = get_bonds_molblock(start)
    else:
        bond_start = get_bonds(start)

    bond_end = get_bonds(end)

    return bond_start == bond_end


def find_unique_confs(best_conformers, mol_files, threshold=0.5):
    """ Clustering conformers with RDKit's Butina algorithm
    to find unique conformer from a list of .sdf files
    using either heavy-atom root mean square deviation (RMSD) 
    or heavy-atom torsion fingerprint deviation (TFD) """

    rdkit_mol = next(rdmolfiles.ForwardSDMolSupplier(mol_files[0], sanitize=False, removeHs=True))
    for mol_file in mol_files[1:]:
        mol = next(rdmolfiles.ForwardSDMolSupplier(mol_file, sanitize=False, removeHs=True))
        rdkit_mol.AddConformer(mol.GetConformer(), assignId=True)

    # calculate difference matrix
    diffmat = AllChem.GetConformerRMSMatrix(rdkit_mol, prealigned=False) #threshold=0.5, sanitize=False, load AllChem
    # diffmat = TorsionFingerprints.GetTFDMatrix(rdkit_mol) #threshold=0.01, sanitize=True, load TorsionFingerprints

    # Cluster conformers
    num_confs = rdkit_mol.GetNumConformers()
    clt = Butina.ClusterData(diffmat, num_confs, threshold,
                             isDistData=True, reordering=True)

    # Get unique conformers
    centroid_idx = [c[0] for c in clt] # centroid indexes
    unique_best_conformers = [best_conformers[i] for i in centroid_idx]
    
    return unique_best_conformers


def ConstrainedEmbedMultipleConfs(mol, core, numConfs=10, useTethers=True, coreConfId=-1, randomseed=2342,
                     getForceField=AllChem.UFFGetMoleculeForceField, numThreads=1, force_constant=1e3):
    match = mol.GetSubstructMatch(core)
    if not match:
        print("molecule doesn't match the core")
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI

    if "." in Chem.MolToSmiles(mol):
        cids = AllChem.EmbedMultipleConfs(mol=mol, numConfs=numConfs, randomSeed=randomseed, numThreads=numThreads, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, ETversion=2, useRandomCoords=True)
    else:
        cids = AllChem.EmbedMultipleConfs(mol=mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomseed, numThreads=numThreads, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, ETversion=2, useRandomCoords=True)

    cids = list(cids)
    if len(cids) == 0:
        print(f'Could not embed molecule.')
        raise ValueError('Could not embed molecule.')

    algMap = [(j, i) for i, j in enumerate(match)]
    
    if not useTethers:
        # clean up the conformation
        for cid in cids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, force_constant)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            rms = AllChem.AlignMol(mol, core, atomMap=algMap)
    else:
        # rotate the embedded conformation onto the core:
        for cid in cids:
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid, ignoreInterfragInteractions=False)
            conf = core.GetConformer()
            for i in range(core.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(pIdx, match[i], 0, 0, force_constant)
            ff.Initialize()
            n = 4
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            while more and n:
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                n -= 1
            # realign
            rms = AllChem.AlignMol(mol, core, prbCid=cid, atomMap=algMap)

    # Remove conformer duplicates
    diffmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False) # calculate difference matrix
    clt = Butina.ClusterData(diffmat, mol.GetNumConformers(), 0.5, isDistData=True, reordering=True) # cluster conformers
    mol2 = Chem.Mol(mol)
    centroids = [mol.GetConformer(id=c[0]) for c in clt] # obtain centroid conformers
    mol2.RemoveAllConformers()
    for c in centroids:
        if len(c.GetPositions()):
            mol2.AddConformer(c, assignId=True)
    
    return mol2


def getAttachmentVector(mol):
    """ Search for the position of the attachment point and extract the atom index of the attachment point and the connected atom (only single neighbour supported)
    Function from https://pschmidtke.github.io/blog/rdkit/3d-editor/2021/01/23/grafting-fragments.html
    mol: rdkit molecule with a dummy atom
    return: atom indices
    """

    rindex = -1
    rindexNeighbor = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            rindex = atom.GetIdx()
            neighbours = atom.GetNeighbors()
            if len(neighbours) == 1:
                rindexNeighbor = neighbours[0].GetIdx()
            else:
                print("two attachment points not supported yet")
                return None
    
    return rindex, rindexNeighbor


def replaceAtom(mol, indexAtom, indexNeighbor, atom_type='Br'):
    """ Replace an atom with another type """
    
    emol = Chem.EditableMol(mol)
    emol.ReplaceAtom(indexAtom, Chem.Atom(atom_type))
    emol.RemoveBond(indexAtom, indexNeighbor)
    emol.AddBond(indexAtom, indexNeighbor, order=Chem.rdchem.BondType.SINGLE)

    return emol.GetMol()


def replaceSelenium(mol, selenium_indicies, replace_Se=True):
    """ selenium_indicies = list of (indexAtom, indexNeighbor1, indexNeighbor2)
        replace_Se = True:  Replace Selenium with Sulfur.
        replace_Se = False: Reintroduce Selenium.
    """
    
    if replace_Se:
        rd_atom = Chem.Atom('S')
    else:
        rd_atom = Chem.Atom('Se')
    
    emol = Chem.EditableMol(mol)

    selenium_indicies = np.array(list(map(np.array, selenium_indicies)))
    for indexAtom, indexNeighbor1, indexNeighbor2 in zip(selenium_indicies[:,0], selenium_indicies[:,1],selenium_indicies[:,2]):
        emol.ReplaceAtom(int(indexAtom), rd_atom)
        emol.RemoveBond(int(indexAtom), int(indexNeighbor1))
        emol.RemoveBond(int(indexAtom), int(indexNeighbor2))
        emol.AddBond(int(indexAtom), int(indexNeighbor1), order=Chem.rdchem.BondType.SINGLE)
        emol.AddBond(int(indexAtom), int(indexNeighbor2), order=Chem.rdchem.BondType.SINGLE)

    return emol.GetMol()


def run_rxn(reactants, smarts): 
    
    # Run reaction
    rxn = AllChem.ReactionFromSmarts(smarts)
    ps = rxn.RunReactants(reactants)
  
    # Find possible products
    product_mols = []
    product_smis = []
    if ps:
        ps = np.asarray(ps)[:,0] # Keep only first product of reaction
        #ps = np.concatenate(ps) # Concatenate all products
    else:
        return product_mols

    for mol in ps:
        # Canonicalize SMILES
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except:
            continue
        
        # Keep only unique products
        if smi not in product_smis:
            product_smis.append(smi)
            mol = Chem.MolFromSmiles(smi)
            Chem.rdmolops.FindPotentialStereoBonds(mol) # finds bonds that could be cis/trans in a molecule and mark them as Bond::STEREOANY
            product_mols.append(mol)
    
    return product_mols


if __name__ == "__main__":
    
    import sys

    print(compare_sdf_structure(sys.argv[1], sys.argv[2]))
