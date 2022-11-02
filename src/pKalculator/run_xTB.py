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

import os
import numpy as np
import shutil
import subprocess

from rdkit import Chem
from ase.units import Hartree, mol, kcal, kJ

import heckqm.molecule_formats as molfmt

# xTB path and calc setup
path = os.getcwd()
XTBHOME = os.path.join(path, 'dep/xtb-6.4.1')
XTBPATH = os.path.join(path, 'dep/xtb-6.4.1/share/xtb')
MANPATH = os.path.join(path, 'dep/xtb-6.4.1/share/man')
LD_LIBRARY_PATH = os.path.join(path, 'dep/xtb-6.4.1/lib')

# XTBHOME = '/groups/kemi/ree/PhD/applications/HeckQM/dep/xtb-6.4.1'
# XTBPATH = '/groups/kemi/ree/PhD/applications/HeckQM/dep/xtb-6.4.1/share/xtb'
# MANPATH = '/groups/kemi/ree/PhD/applications/HeckQM/dep/xtb-6.4.1/share/man'
# LD_LIBRARY_PATH = '/groups/kemi/ree/PhD/applications/HeckQM/dep/xtb-6.4.1/lib'

OMP_NUM_THREADS = '1'
MKL_NUM_THREADS = '1'


def run_xTB(args): #(xyzfile, molecule, chrg=0, spin=0, method=' 1', solvent='', optimize=True, precalc_path=None):

    global XTBHOME
    global XTBPATH
    global LD_LIBRARY_PATH

    global OMP_NUM_THREADS
    global MKL_NUM_THREADS

    # Set env parameters for xTB
    os.environ['XTBHOME'] = XTBHOME
    os.environ['XTBPATH'] = XTBPATH
    os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ['MKL_NUM_THREADS'] = MKL_NUM_THREADS

    # Unpack inputs
    xyzfile, molecule, chrg, spin, method, solvent, optimize, precalc_path = args

    # Create calculation directory
    name = xyzfile.split('.')[0]
    mol_calc_path = os.path.join(os.getcwd(), 'calc', name.split('_')[0], name.split('_')[1], 'full_opt', name)
    os.makedirs(mol_calc_path, exist_ok=True)
    
    # Create files in calculation directory
    start_structure_xyz = os.path.join(os.getcwd(), 'calc', name.split('_')[0], name.split('_')[1], xyzfile)
    start_structure_sdf = os.path.join(mol_calc_path, name+'.sdf')
    final_structure_sdf = os.path.join(mol_calc_path, name+'_opt.sdf')
    if precalc_path:
        shutil.copy(os.path.join(precalc_path, 'xtbopt.xyz'), start_structure_xyz) #copy xyz file of molecule from precalc_path
    else:
        Chem.rdmolfiles.MolToXYZFile(molecule, start_structure_xyz) #make xyz file of molecule (without isotope information)

    # Run xTB calc
    if optimize:
        cmd = f'{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --lmo --chrg {chrg} --uhf {spin} {solvent}'
    else:
        cmd = f'{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --lmo --chrg {chrg} --uhf {spin} {solvent}'

    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=mol_calc_path)
    output = proc.communicate()[0]

    # Save calc output
    with open(f'{mol_calc_path}/{name}.xtbout', 'w') as f:
        f.write(output)

    # Convert .xyz to .sdf using and check input/output connectivity
    if os.path.isfile(f'{mol_calc_path}/xtbopt.xyz'):
        molfmt.convert_xyz_to_sdf(start_structure_xyz, start_structure_sdf) #convert initial structure
        molfmt.convert_xyz_to_sdf(f'{mol_calc_path}/xtbopt.xyz', final_structure_sdf) #convert optimized structure
    else:
        print(f'WARNING! xtbopt.xyz was not created => calc failed for {xyzfile}')
        energy = 60000.0
        return energy, mol_calc_path
    
    same_structure = molfmt.compare_sdf_structure(Chem.MolToMolBlock(molecule), final_structure_sdf, molblockStart=True)
    if not same_structure:
        print(f'WARNING! Input/output mismatch for {xyzfile}')
        energy = 60000.0
        return energy, mol_calc_path

    # Search for the molecular energy
    for i, line in enumerate(output.split('\n')):
        if 'TOTAL ENERGY' in line:
            energy = line.split()[3]

    try: #check if the molecular energy was found.
        energy = float(energy) * Hartree * mol/kJ #convert energy from Hartree to kJ/mol #* mol/kcal #convert energy from Hartree to kcal/mol
    except Exception as e:
        print(e, xyzfile)
        energy = 60000.0

    return energy, mol_calc_path


def make_xcontrol_contrain_file(molecule, core, path):
    
    # Locate atoms to contrain
    match = np.array(molecule.GetSubstructMatch(core)) + 1 # indexing starts with 0 for RDKit but 1 for xTB
    match = sorted(match)
    assert len(match) == core.GetNumAtoms(), 'ERROR! Complete match not found.'

    # Write the xcontrol file
    fo = open(os.path.join(path, 'xcontrol'), 'w')
    fo.write('$constrain\n')
    fo.write(' force constant=0.5\n')
    fo.write(f' atoms: {",".join(map(str, match))}\n')
    fo.write('$end\n')
    fo.close()

    return


def run_contrained_xTB(args): #(molecule, core, name, chrg=0, spin=0, method=' 1', solvent='', precalc_path=None):

    global XTBHOME
    global XTBPATH
    global LD_LIBRARY_PATH

    global OMP_NUM_THREADS
    global MKL_NUM_THREADS

    # Set env parameters for xTB
    os.environ['XTBHOME'] = XTBHOME
    os.environ['XTBPATH'] = XTBPATH
    os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ['MKL_NUM_THREADS'] = MKL_NUM_THREADS

    # Unpack inputs
    molecule, core, name, chrg, spin, method, solvent, precalc_path = args

    # Create calculation directory
    mol_calc_path = os.path.join(os.getcwd(), 'calc', name.split('_')[0], name.split('_')[1], 'contrained_'+name.split('_')[-1], name)
    os.makedirs(mol_calc_path, exist_ok=True)
    
    # Create files in calculation directory
    start_structure_xyz = os.path.join(mol_calc_path, f'{name}.xyz')
    final_structure_sdf = os.path.join(mol_calc_path, f'{name}_opt.sdf')
    if precalc_path:
        shutil.copy(os.path.join(precalc_path, 'xtbopt.xyz'), start_structure_xyz) #copy xyz file of molecule from precalc_path
        shutil.copy(os.path.join(precalc_path, 'xcontrol'), os.path.join(mol_calc_path, 'xcontrol')) #copy xcontrol from precalc_path
    else:
        Chem.rdmolfiles.MolToXYZFile(molecule, start_structure_xyz) #make xyz file of molecule (without isotope information)
        make_xcontrol_contrain_file(molecule, core, mol_calc_path) #make xcontrol file

    # Run xTB calc
    cmd = f'{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --chrg {chrg} --uhf {spin} --input ./xcontrol {solvent}'
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=mol_calc_path)
    output = proc.communicate()[0]

    # Save calc output
    with open(f'{mol_calc_path}/{name}.xtbout', 'w') as f:
        f.write(output)

    # Convert .xyz to .sdf using and check input/output connectivity
    if os.path.isfile(f'{mol_calc_path}/xtbopt.xyz'):
        molfmt.convert_xyz_to_sdf(f'{mol_calc_path}/xtbopt.xyz', final_structure_sdf) #convert optimized structure
    else:
        print(f'WARNING! xtbopt.xyz was not created => calc failed for {name}')
        energy = 60000.0
        return energy, mol_calc_path
    
    same_structure = molfmt.compare_sdf_structure(Chem.MolToMolBlock(molecule), final_structure_sdf, molblockStart=True)
    if not same_structure:
        print(f'WARNING! Input/output mismatch for {name}')
        energy = 60000.0
        return energy, mol_calc_path

    # Search for the molecular energy
    for i, line in enumerate(output.split('\n')):
        if 'TOTAL ENERGY' in line:
            energy = line.split()[3]

    try: #check if the molecular energy was found.
        energy = float(energy) * Hartree * mol/kJ #convert energy from Hartree to kJ/mol
    except Exception as e:
        print(e, name)
        energy = 60000.0

    return energy, mol_calc_path
