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
from ase.units import Hartree, mol, kcal
import molecule_formats as molfmt
from pathlib import Path
import re


def load_and_prepare_xyz(path, xyz_file):
    input_file = np.loadtxt(fname=f"{path}/{xyz_file}", skiprows=2, dtype=str, ndmin=2)
    xyz_coords = []
    for line in input_file:
        xyz_coords.append(
            f"{line[0]:4} {float(line[1]):11.6f} {float(line[2]):11.6f} {float(line[3]):11.6f}\n"
        )
    return xyz_coords


def get_latest_geom(file: str) -> list:
    """A function to get the latest geometry xyz coordinated from an Orca output file

    regular expression pattern:
    `\s*`: zero or more whitespace characters at the beginning of the line
    `[A-Z][a-z]?`: an element symbol (one or two characters,
    with the first character uppercase and the second character lowercase)
    `\s+`: one or more whitespace characters
    `-?\d+\.\d+`: a floating-point number (positive or negative)
    `\s+`: one or more whitespace characters
    `-?\d+\.\d+`: a floating-point number (positive or negative)
    `\s+`: one or more whitespace characters
    `-?\d+\.\d+`: a floating-point number (positive or negative)
    `\n`: a newline character
    Args:
        file (str): file path for the Orca output file

    Returns:
        list: of xyz coordinates for the latest geometry optimization step
    """
    # Initialize variables
    record_lines = False
    lst_lines = []
    new_list = []
    temp_list = []

    # Open file and read line by line
    with open(Path(file), "r") as f:
        for line in f:
            # Check if we should start recording lines
            if not record_lines:
                if line.startswith("CARTESIAN COORDINATES (ANGSTROEM)"):
                    # if line.startswith("VIBRATIONAL FREQUENCIES"):
                    record_lines = True
            # Check if we should stop recording lines
            elif line.startswith("CARTESIAN COORDINATES (A.U.)"):
                record_lines = False
            # Append recorded lines
            else:
                lst_lines.append(line)
            # lst_lines_per_opt.append(lst_lines)

    # Define regular expression pattern to match table start
    table_pattern = re.compile(
        r"\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\n"
    )
    # Match pattern in each line and store in a list
    lst_match = [re.match(table_pattern, line) for line in lst_lines]

    # Group matches into lists
    for match in lst_match:
        if match is not None:
            temp_list.append(match.group().split())
        elif temp_list:
            new_list.append(temp_list)
            temp_list = []

    # Add last group of matches to new_list
    if temp_list:
        new_list.append(temp_list)

    # Convert list of lists to list of tab-separated strings
    lst_latest_geom = ["\t".join(line) + "\n" for line in new_list[-1]]

    return lst_latest_geom


def create_orca_input(
    xyz_file,
    chrg,
    path,
    ncores=2,
    mem=10000,
    functional="",
    basis_set="",
    dispersion="",
    opt=False,
    freq=False,
    solvent_name="DMSO",
    inp_name="orca_calc.inp",
):
    with open(Path(f"{path}/{inp_name}"), "w") as input_file:
        input_file.write("# ORCA input file\n")

        base_str = f"! {functional}"
        if dispersion:
            base_str += " D4"
        base_str += f" {basis_set}"

        if functional.lower() == "r2scan-3c":
            base_str = "! R2SCAN-3C"

        if opt:
            base_str += " OPT"
        if freq:
            base_str += " FREQ"
        if opt or freq:
            base_str += f" CPCM({solvent_name})"
        else:
            base_str += " CPCM"
        base_str += "\n"

        input_file.write(base_str)

        input_file.write(
            f"\n%maxcore {(mem * 0.75) / ncores}\n%pal nprocs {ncores} end\n"
        )
        input_file.write("\n%scf\nMaxIter 1000\nend\n")

        if not (opt or freq):  # Adjust condition for CPCM
            input_file.write(f'%cpcm smd true SMDsolvent "{solvent_name}" end\n')

        input_file.write(
            f'\n*xyz {chrg} 1\n{"".join(load_and_prepare_xyz(path=path, xyz_file=xyz_file))}*'
        )


def create_orca_input_rerun(
    chrg,
    path,
    ncores=2,
    mem=10000,
    functional="",
    basis_set="",
    dispersion="",
    opt=False,
    freq=False,
    solvent_name="DMSO",
    inp_name="orca_calc_rerun.inp",
):
    lst_latest_geom = None
    loaded_xyz_file = None  # Ensure this is defined for all paths
    orca_calc_xyz = "orca_calc.xyz"
    xtbopt_xyz = "xtbopt.xyz"

    if Path(Path(path) / "orca_calc.out").is_file():
        same_structure = check_input_output_structure_orca(
            path=path, xyzfile=orca_calc_xyz
        )

        if not same_structure:
            if Path(Path(path) / xtbopt_xyz).is_file():
                loaded_xyz_file = load_and_prepare_xyz(path=path, xyz_file=xtbopt_xyz)
                print("using xtbopt.xyz to create 'orca_calc_rerun.inp'")
        elif same_structure and opt:
            lst_latest_geom = get_latest_geom(Path(path) / "orca_calc.out")
            print(
                "using latest geometry optimization from orca_calc.out to create 'orca_calc_rerun.inp'"
            )
        elif Path(Path(path) / orca_calc_xyz).is_file():
            loaded_xyz_file = load_and_prepare_xyz(path=path, xyz_file=orca_calc_xyz)
            print("using orca_calc.xyz to create 'orca_calc_rerun.inp'")
    else:
        raise Exception(f"No .out or .xyz file found in {path}")

    with open(Path(f"{path}/{inp_name}"), "w") as input_file:
        input_file.write("# ORCA input file\n")

        # Adjusting command line based on functional, opt, freq
        base_str = f"! {functional}"
        if dispersion:
            base_str += " D4"
        base_str += f" {basis_set} Slowconv"

        if functional.lower() == "r2scan-3c":
            base_str = "!R2SCAN-3C Slowconv"

        if opt:
            base_str += " OPT"
        if freq:
            base_str += " FREQ"
        if opt or freq:
            base_str += f" CPCM({solvent_name})"
        else:
            base_str += " CPCM"
        base_str += "\n"

        input_file.write(base_str)
        input_file.write(
            f"\n%maxcore {(mem * 0.75) / ncores}\n%pal nprocs {ncores} end\n"
        )
        if not (opt or freq):
            input_file.write(f'%cpcm smd true SMDsolvent "{solvent_name}" end\n')
        input_file.write("\n%scf\nMaxIter 5000\nend\n")
        input_file.write("\n%geom\nMaxIter 5000\nMaxStep 0.1\nend\n")

        # Handling XYZ content
        input_file.write(f"\n*xyz {chrg} 1\n")
        if lst_latest_geom is not None:
            input_file.writelines(lst_latest_geom)
        elif loaded_xyz_file is not None:
            input_file.writelines("".join(loaded_xyz_file))
        input_file.write("*")

    print(f"{inp_name} created")


def run_orca_calculation(path, input_file="orca_calc.inp"):
    # Run ORCA calc here
    # Change the paths under the cmd variable to match your system
    cmd = f'env - PATH="/groups/kemi/borup/pKalculator/dep/orca_5_0_4:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH" LD_LIBRARY_PATH="/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH" /bin/bash -c "/groups/kemi/borup/pKalculator/dep/orca_5_0_4/orca {path}/{input_file}"'

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=f"{path}",
    )
    output = proc.communicate()[0]
    return output


def run_orca(
    xyz_file,
    chrg,
    path,
    ncores=2,
    mem=10000,
    functional="",
    basis_set="",
    dispersion="",
    opt=False,
    freq=False,
    solvent_name="DMSO",
):
    # Create ORCA input file
    create_orca_input(
        xyz_file=xyz_file,
        chrg=chrg,
        path=path,
        ncores=ncores,
        mem=mem,
        functional=functional,
        basis_set=basis_set,
        dispersion=dispersion,
        opt=opt,
        freq=freq,
        solvent_name=solvent_name,
        inp_name="orca_calc.inp",
    )

    # Run ORCA calc
    print("starting ORCA calculation")
    output = run_orca_calculation(path=path, input_file="orca_calc.inp")

    # Save calc output
    save_orca_output(output=output, path=path, output_file="orca_calc.out")

    # Check if ORCA calc was successful
    energy = check_orca_success(path=path, opt=opt, out_file="orca_calc")

    return energy


def rerun_orca(
    chrg,
    path,
    ncores=2,
    mem=10000,
    functional="",
    basis_set="",
    dispersion="",
    opt=False,
    freq=False,
    solvent_name="DMSO",
):
    """
    Rerun ORCA calculation with new input file
    """
    create_orca_input_rerun(
        chrg=chrg,
        path=path,
        ncores=ncores,
        mem=mem,
        functional=functional,
        basis_set=basis_set,
        dispersion=dispersion,
        opt=opt,
        freq=freq,
        solvent_name=solvent_name,
        inp_name="orca_calc_rerun.inp",
    )

    print("starting re-run of ORCA calculation")
    # Run ORCA calc
    output = run_orca_calculation(path=path, input_file="orca_calc_rerun.inp")

    # Save calc output
    save_orca_output(output=output, path=path, output_file="orca_calc_rerun.out")

    # Check if ORCA calc was successful
    energy = check_orca_success(path=path, opt=opt, out_file="orca_calc_rerun")

    return energy


def run_orca_multiple(
    xyz_file,
    chrg,
    path,
    ncores=2,
    mem=10000,
    functional="",
    basis_set="",
    dispersion="",
    optfreq=False,
    solvent_name="DMSO",
):
    """NOT IMPLEMENTED YET"""
    # Multiple jobs
    input_file = open(f"{path}/orca_calc.inp", "w")

    input_file.write("# ORCA input file\n")

    # 1st job
    input_file.write(f"!R2SCAN-3C OPT FREQ CPCM({solvent_name}) xyzfile\n")
    input_file.write('%base "r2scan-3c-optfreq"\n')
    input_file.write(f"\n%maxcore {(mem * 0.75) / ncores}\n%pal nprocs {ncores} end\n")

    input_file.write(f"*xyz {chrg} 1 \n")
    input_file.write(
        f'\n*xyz {chrg} 1\n{"".join(load_and_prepare_xyz(path=path, xyz_file=xyz_file))}*'
    )

    # 2nd job
    input_file.write("\n$new_job\n")
    input_file.write("!SP CAM-B3LYP D4 def2-TZVPPD CPCM\n")
    input_file.write('%base "cam-b3lyp-d4-sp"\n')
    input_file.write(f"\n%maxcore {(mem * 0.75) / ncores}\n%pal nprocs {ncores} end\n")
    input_file.write(f'%cpcm smd true SMDsolvent "{solvent_name}" end\n')
    input_file.write(f"*xyz {chrg} 1 \n")
    input_file.write(
        f'\n*xyz {chrg} 1\n{"".join(load_and_prepare_xyz(path=path, xyz_file="r2scan-3c-optfreq.xyz"))}*'
    )
    input_file.close()

    # Run ORCA calc
    output = run_orca_calculation(path=path, input_file="orca_calc.inp")

    # Save calc output
    with open(f"{path}/r2scan-3c-optfreq.out", "w") as f:
        f.write(output)

    # Check if ORCA calc was successful
    if not check_input_output_structure_orca(
        path=path, xyz_file="r2scan-3c-optfreq.xyz"
    ):
        energy = float("inf")
    else:
        energy = get_energy_orca(path=path, out_file="r2scan-3c-optfreq.out")
        print(f"energy for optfreq: {energy}")
        print(get_frequencies_orca(path=path, out_file="r2scan-3c-optfreq.out"))

    energy = get_energy_orca(path=path, out_file="cam-b3lyp-d4-sp.out")
    print(f"energy for sp: {energy}")

    return energy


# -------------------------
# Post ORCA run functions
# -------------------------


def save_orca_output(output, path, output_file="orca_calc.out"):
    # Save calc output here
    with open(f"{path}/{output_file}", "w") as f:
        f.write(output)


def check_orca_success(path, opt=False, out_file="orca_calc"):
    # Check if ORCA calc was successful here
    if opt:
        if not check_input_output_structure_orca(path=path, xyzfile=f"{out_file}.xyz"):
            energy = float("inf")
        else:
            energy = get_energy_orca(path=path, out_file=f"{out_file}.out")
            print(get_frequencies_orca(path=path, out_file=f"{out_file}.out"))
    else:
        energy = get_energy_orca(path=path, out_file=f"{out_file}.out")
    return energy


def check_input_output_structure_orca(path, xyzfile="orca_calc.xyz"):
    # if optimization has been performed
    # check if the input and output structures are the same
    xyzfile.split(".")[0]
    final_structure_sdf = Path(path).joinpath(f'{xyzfile.split(".")[0]}.sdf')

    if Path(Path(path) / f"{xyzfile}").is_file():
        for sdf in Path(path).glob("*.sdf"):
            if "_opt" not in sdf.name:
                start_structure_sdf = sdf
        molfmt.convert_xyz_to_sdf(Path(Path(path) / f"{xyzfile}"), final_structure_sdf)
        same_structure = molfmt.compare_sdf_structure(
            start_structure_sdf, final_structure_sdf, molblockStart=False
        )
        if not same_structure:
            print(
                f"WARNING! Input/output mismatch for {xyzfile} after ORCA optimization."
            )
        return same_structure


def get_energy_orca(path, out_file="orca_calc.out"):
    convergence_idx = 0
    freq = False
    error_occurred = False
    energy = float("inf")

    with open(Path(Path(path) / out_file), "r") as f:
        output_lines = f.read().splitlines()

    for line in output_lines:
        if (
            "The optimization did not converge but reached the maximum" in line
            or "ORCA finished by error termination" in line
        ):
            print("WARNING: Error in ORCA calculation")
            error_occurred = True
            break
        if "FREQ" in line:
            freq = True
        if "THE OPTIMIZATION HAS CONVERGED" in line:
            convergence_idx = output_lines.index(line)

    if not error_occurred and convergence_idx != 0:
        for line in output_lines[convergence_idx + 1 :]:
            if (
                "Check your MOs and check whether a frozen core calculation is appropriate"
                in line
                or "**** WARNING: LOEWDIN FINDS" in line
                or "**** WARNING: MULLIKEN FINDS" in line
            ):
                print("WARNING: Error in ORCA calculation")
                break

            if (
                not freq
                and "FINAL SINGLE POINT ENERGY" in line
                and (
                    "(Wavefunction not fully converged!)" not in line
                    and "(SCF not fully converged!)" not in line
                )
            ):
                energy = float(line.split()[-1]) * Hartree * mol / kcal
                break
            if "Total thermal energy" in line:  # for FREQ calculations
                energy = float(line.split()[-2]) * Hartree * mol / kcal
                break

    elif not error_occurred and convergence_idx == 0:
        # for single point calculations
        for line in output_lines[convergence_idx + 1 :]:
            if (
                "Check your MOs and check whether a frozen core calculation is appropriate"
                in line
                or "**** WARNING: LOEWDIN FINDS" in line
                or "**** WARNING: MULLIKEN FINDS" in line
            ):
                print("WARNING: Error in ORCA calculation")
                break

            if (
                not freq
                and "FINAL SINGLE POINT ENERGY" in line
                and "(Wavefunction not fully converged!)" not in line
            ):
                energy = float(line.split()[-1]) * Hartree * mol / kcal
                break
            if "Total thermal energy" in line:  # for FREQ calculations
                energy = float(line.split()[-2]) * Hartree * mol / kcal
                break

    return energy


def get_frequencies_orca(path, out_file="orca_calc.out"):
    # Get frequencies from output
    record_lines = False
    lst_lines = []
    with open(Path(Path(path) / out_file), "r") as f:
        for i, line in enumerate(f):
            if not record_lines:
                if line.startswith("Scaling factor for frequencies"):
                    # if line.startswith("VIBRATIONAL FREQUENCIES"):
                    record_lines = True
            elif line.startswith("NORMAL MODES"):
                record_lines = False
            else:
                lst_lines.append(line)

    reg_compile = re.compile(r"\d+\.\d+")  # finds all floats in a string
    freq = [
        float(re.search(reg_compile, line).group())
        for line in lst_lines
        if re.search(reg_compile, line) is not None
    ]
    neg_freq = sum(n < 0 for n in freq)
    if neg_freq > 0:
        print(f"WARNING! {neg_freq} negative frequencies found")
    print("Frequencies:")
    return freq
