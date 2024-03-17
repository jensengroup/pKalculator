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
import re
from pathlib import Path
from rdkit import Chem
from ase.units import Hartree, mol, kcal, kJ

import molecule_formats as molfmt


def extract_version(dir_name):
    # Function to extract version numbers from directory names
    match = re.findall(r"\d+", dir_name)
    return tuple(map(int, match)) if match else (0,)


# Go to the base directory pKalculator
base_dir = Path(__file__).resolve().parent.parent
# Filter out the directories that contain "xtb-"
xtb_dirs = [
    item
    for item in base_dir.joinpath("dep").iterdir()
    if "xtb-" in item.name and item.is_dir()
]
# Sort the directories by version number
xtb_dirs_sorted = sorted(
    xtb_dirs, key=lambda dir: extract_version(dir.name), reverse=True
)
# The directory with the highest version number is the first one in the sorted list
highest_version_xtb = xtb_dirs_sorted[0] if xtb_dirs_sorted else None


# xTB path and calc setup
base_dir = str(base_dir)
XTBHOME = str(highest_version_xtb)
XTBPATH = str(highest_version_xtb.joinpath("share/xtb"))
MANPATH = str(highest_version_xtb.joinpath("share/man"))
LD_LIBRARY_PATH = str(highest_version_xtb.joinpath("lib"))

OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"


def run_xTB(args):
    # (xyzfile, molecule, chrg=0, spin=0, method=' 1', solvent='', optimize=True, precalc_path=None):

    global XTBHOME
    global XTBPATH
    global LD_LIBRARY_PATH

    global OMP_NUM_THREADS
    global MKL_NUM_THREADS

    # Set env parameters for xTB
    os.environ["XTBHOME"] = XTBHOME
    os.environ["XTBPATH"] = XTBPATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS

    # Unpack inputs
    # xyzfile, molecule, chrg, spin, method, solvent, optimize, precalc_path = args
    (
        xyzfile,
        molecule,
        chrg,
        spin,
        method,
        solvent,
        optimize,
        precalc_path,
        calc_path,
    ) = args

    # Create calculation directory
    name = xyzfile.split(".")[0]
    mol_calc_path = os.path.join(
        str(base_dir),
        calc_path,
        name.split("_")[0],
        name.split("_")[1],
        name.split("_")[-1],
    )
    os.makedirs(mol_calc_path, exist_ok=True)

    # Create files in calculation directory
    start_structure_xyz = os.path.join(
        str(base_dir), calc_path, name.split("_")[0], name.split("_")[1], xyzfile
    )
    start_structure_sdf = os.path.join(mol_calc_path, name + ".sdf")
    final_structure_sdf = os.path.join(mol_calc_path, name + "_opt.sdf")
    if precalc_path:
        shutil.copy(
            os.path.join(precalc_path, "xtbopt.xyz"), start_structure_xyz
        )  # copy xyz file of molecule from precalc_path
    else:
        Chem.rdmolfiles.MolToXYZFile(
            molecule, start_structure_xyz
        )  # make xyz file of molecule (without isotope information)

    # Run xTB calc
    # if optimize:
    #     cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --lmo --chrg {chrg} --uhf {spin} {solvent}"
    # else:
    #     cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --lmo --chrg {chrg} --uhf {spin} {solvent}"

    # proc = subprocess.Popen(
    #     cmd.split(),
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.DEVNULL,
    #     text=True,
    #     cwd=mol_calc_path,
    # )
    # output = proc.communicate()[0]

    if optimize:
        cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --lmo --chrg {chrg} --uhf {spin} {solvent}"
        try:
            cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --opt --lmo --chrg {chrg} --uhf {spin} {solvent}"
            proc = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=mol_calc_path,
            )
            output = proc.communicate()[0]
        except subprocess.CalledProcessError:
            try:
                cmd = f"xtb --gfn{method} {start_structure_xyz} --opt --lmo --chrg {chrg} --uhf {spin} {solvent}"
                proc = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    cwd=mol_calc_path,
                )
                output = proc.communicate()[0]
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                return None
    else:
        cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --lmo --chrg {chrg} --uhf {spin} {solvent}"
        try:
            cmd = f"{XTBHOME}/bin/xtb --gfn{method} {start_structure_xyz} --lmo --chrg {chrg} --uhf {spin} {solvent}"
            proc = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=mol_calc_path,
            )
            output = proc.communicate()[0]
        except subprocess.CalledProcessError:
            try:
                cmd = f"xtb --gfn{method} {start_structure_xyz} --lmo --chrg {chrg} --uhf {spin} {solvent}"
                proc = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    cwd=mol_calc_path,
                )
                output = proc.communicate()[0]
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                return None

    # Save calc output
    with open(f"{mol_calc_path}/{name}.xtbout", "w") as f:
        f.write(output)

    # Convert .xyz to .sdf using and check input/output connectivity
    if os.path.isfile(f"{mol_calc_path}/xtbopt.xyz"):
        molfmt.convert_xyz_to_sdf(
            start_structure_xyz, start_structure_sdf
        )  # convert initial structure
        molfmt.convert_xyz_to_sdf(
            f"{mol_calc_path}/xtbopt.xyz", final_structure_sdf
        )  # convert optimized structure
        same_structure = molfmt.compare_sdf_structure(
            Chem.MolToMolBlock(molecule), final_structure_sdf, molblockStart=True
        )
        if not same_structure:
            print(f"WARNING! Input/output mismatch for {xyzfile}")
            energy = float(99999)
            return energy, mol_calc_path
    else:
        print(f"WARNING! xtbopt.xyz was not created => calc failed for {xyzfile}")
        energy = float(99999)
        return energy, mol_calc_path

    # Search for the molecular energy
    for i, line in enumerate(output.split("\n")):
        if "TOTAL ENERGY" in line:
            energy = line.split()[3]

    try:  # check if the molecular energy was found.
        energy = (
            float(energy) * Hartree * mol / kcal
        )  # convert energy from Hartree to kcal/mol
    except Exception as e:
        print(e, xyzfile)
        energy = float(99999)

    return energy, mol_calc_path


def get_energy_xtb(path, out_file):
    energy = float(99999)
    with open(Path(Path(path) / out_file), "r") as f:
        output_lines = f.read().splitlines()

        for line in reversed(output_lines):
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[3])
                energy = energy * Hartree * mol / kcal
                break
    return energy
