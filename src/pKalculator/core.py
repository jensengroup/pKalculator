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
from pathlib import Path
import argparse
import copy
import numpy as np
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

import molecule_formats as molfmt
import run_xTB as run_xTB
from run_orca import run_orca
from run_orca import rerun_orca
from modify_smiles import deprotonate

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# CPU and memory usage
# num_cpu_single = 12  # 24 # number of cpus per job.
# mem_gb = 24  # 30 # total memory usage per task.
num_cpu_single = 10  # 24 # number of cpus per job.
mem_gb = 20  # 30 # total memory usage per task.


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Get deprotonated energy values of a molecule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-csv",
        "--csv_path",
        metavar="csv_path",
        help="csv path. The csv file must be comma seperated and contain a column named'smiles' and a column named 'names'",
        type=str,
    )

    parser.add_argument(
        "-calc",
        "--calc_path",
        metavar="calc_path",
        help="calculation path",
        type=str,
    )

    parser.add_argument(
        "-submitit",
        "--submitit_path",
        metavar="submitit_path",
        help="submitit path",
        type=str,
    )

    parser.add_argument(
        "-f", "--functional", metavar="functional", help="functional", type=str
    )

    parser.add_argument("-b", "--basis", metavar="basis", help="basis set", type=str)

    parser.add_argument(
        "-s",
        "--solvent",
        metavar="solvent",
        help="Enter solvent name. Default is DMSO",
        default="DMSO",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dispersion",
        help="set dispersion",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--optfreq",
        help="set optfreq",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    return args


def confsearch_xTB(
    conf_complex_mols,
    conf_names,
    chrg=0,
    spin=0,
    method="ff",
    solvent="",
    conf_cutoff=10,
    precalc_path=None,
):
    global num_cpu_single
    global calc_path

    # Run a constrained xTB optimizations
    confsearch_args = []
    for i in range(len(conf_names)):
        if precalc_path:
            confsearch_args.append(
                (
                    conf_names[i] + "_gfn" + method.replace(" ", "") + ".xyz",
                    conf_complex_mols[i],
                    chrg,
                    spin,
                    method,
                    solvent,
                    True,
                    precalc_path[i],
                    calc_path,
                )
            )
        else:
            confsearch_args.append(
                (
                    conf_names[i] + "_gfn" + method.replace(" ", "") + ".xyz",
                    conf_complex_mols[i],
                    chrg,
                    spin,
                    method,
                    solvent,
                    True,
                    None,
                    calc_path,
                )
            )

    with ThreadPoolExecutor(max_workers=num_cpu_single) as executor:
        results = executor.map(run_xTB.run_xTB, confsearch_args)

    conf_energies = []
    conf_paths = []
    for result in results:
        conf_energy, path_opt = result
        conf_energies.append(conf_energy)
        conf_paths.append(path_opt)

    # Find the conformers below cutoff #kcal/mol (3 kcal/mol = 12.6 kJ/mol)
    # Note conf energy cannot be inf for this to work
    # covert to relative energies
    rel_conf_energies = np.array(conf_energies) - np.min(conf_energies)

    # get number of conf below cutoff
    below_cutoff = (rel_conf_energies <= conf_cutoff).sum()

    # make a tuple
    conf_tuple = list(
        zip(conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies)
    )

    # get only the best conf below cutoff
    conf_tuple = sorted(conf_tuple, key=itemgetter(4))[:below_cutoff]

    # unzip tuple to variables
    conf_names, conf_complex_mols, conf_paths, conf_energies, rel_conf_energies = zip(
        *conf_tuple
    )

    # tuples to lists
    conf_names, conf_complex_mols, conf_paths, conf_energies = (
        list(conf_names),
        list(conf_complex_mols),
        list(conf_paths),
        list(conf_energies),
    )

    # list of paths to optimized structures in .sdf format
    mol_files = [
        os.path.join(
            conf_path, conf_name + "_gfn" + method.replace(" ", "") + "_opt.sdf"
        )
        for conf_path, conf_name in zip(conf_paths, conf_names)
    ]

    # Find only unique conformers
    conf_names, conf_complex_mols, conf_paths, conf_energies = zip(
        *molfmt.find_unique_confs(
            list(zip(conf_names, conf_complex_mols, conf_paths, conf_energies)),
            mol_files,
            threshold=0.5,
        )
    )

    # tuples to lists
    conf_names, conf_complex_mols, conf_paths, conf_energies = (
        list(conf_names),
        list(conf_complex_mols),
        list(conf_paths),
        list(conf_energies),
    )

    return conf_names, conf_complex_mols, conf_paths, conf_energies


def calculateEnergy(rdkit_mol: Chem.rdchem.Mol, name: str, smi: str):
    """_summary_

    Args:
        rdkit_mol (_type_): _description_
        name (_type_): _description_
        smi (_type_): _description_ smi is not used in this function

    Raises:
        Exception: _description_

    Returns:
        _type_: energy xtb, energy sp, best_conf [kcal/mol]
    """

    global calc_path
    global num_cpu_single
    global method, solvent_model, solvent_name

    global functional, basis
    global dispersion
    global optfreq

    # change the method for accurate calculations ('ff', ' 0', ' 1', ' 2')
    # standard: method=' 2'
    method = " " + str(method)
    # change the solvent ('--gbsa solvent_name', '--alpb solvent_name', or '')
    # standard: solvent = 'DMSO'
    solvent = "--" + solvent_model + " " + solvent_name

    # checking and adding formal charge to "chrg"
    chrg = Chem.GetFormalCharge(rdkit_mol)
    # OBS! Spin is hardcoded to zero!

    # RDkit conf generator
    rdkit_mol = Chem.AddHs(rdkit_mol)
    rot_bond = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdkit_mol)
    n_conformers = min(1 + 3 * rot_bond, 20)

    # check if the molecule has a bridgehead atom and add more conformers
    query_bridgehead_mol = Chem.MolFromSmarts("[Ax{3-4}]")  #'[x{3-4}]']
    if rdkit_mol.GetSubstructMatches(query_bridgehead_moll):
        n_conformers = min(10 + 3 * rot_bond, 20)

    ps = AllChem.ETKDGv3()
    ps.useExpTorsionAnglePrefs = True
    ps.useBasicKnowledge = True
    ps.ETversion = 2
    ps.randomSeed = 90
    ps.useSmallRingTorsions = True
    ps.useRandomCoords = False
    ps.maxAttempts = 3
    ps.enforceChirality = True

    cids = AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=n_conformers, params=ps)

    if not cids:
        print(
            f"1st embed failed for {name} with SMILES: {smi}\n"
            "will try useRandomCoords=True"
        )
        ps.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(
            rdkit_mol,
            numConfs=n_conformers,
            params=ps,
        )
        if not cids:
            print(
                f"2nd embed failed for {name} with SMILES: {smi}\n"
                "will try useExpTorsionAnglePref=False and enforceChirality=False"
            )
            ps.useExpTorsionAnglePrefs = False
            ps.enforceChirality = False
            cids = AllChem.EmbedMultipleConfs(
                rdkit_mol,
                numConfs=n_conformers,
                params=ps,
            )
            if not cids:
                print(
                    f"3nd embed failed for {name} with SMILES: {smi}\n"
                    "will try to n_conformers*5 and maxAttempts=1000"
                )
                cids = AllChem.EmbedMultipleConfs(
                    rdkit_mol,
                    numConfs=n_conformers * 5,
                    params=ps,
                    maxAttempts=1000,
                )
                if len(list(cids)) == 0:
                    raise Exception(f"4rd embed failed for {name} with SMILES: {smi}")

    # Unpack confomers and assign conformer names
    conf_mols = [
        Chem.Mol(rdkit_mol, False, i) for i in range(rdkit_mol.GetNumConformers())
    ]

    # change zfill(2) if more than 99 conformers
    conf_names = [
        name + f"_conf{str(i+1).zfill(2)}" for i in range(rdkit_mol.GetNumConformers())
    ]

    conf_names_copy = copy.deepcopy(conf_names)

    # Run a GFN-FF optimization
    conf_names, conf_mols, conf_paths, conf_energies = confsearch_xTB(
        conf_mols,
        conf_names,
        chrg=chrg,
        spin=0,
        method="ff",
        solvent=solvent,
        conf_cutoff=3,
        precalc_path=None,
    )

    # Run a GFN?-xTB optimization
    conf_names, conf_mols, conf_paths, conf_energies = confsearch_xTB(
        conf_mols,
        conf_names,
        chrg=chrg,
        spin=0,
        method=method,
        solvent=solvent,
        conf_cutoff=3,
        precalc_path=conf_paths,
    )

    # Run Orca single point calculations
    # Note conf energy cannot be inf for this to work
    final_conf_energies = []
    final_conf_mols = []
    for conf_name, conf_mol, conf_path, conf_energy in zip(
        conf_names, conf_mols, conf_paths, conf_energies
    ):
        # Uncomment for single point calculations on all unique conformers
        # if conf_energy != float(99999):
        # conf_energy = run_orca.run_orca(
        # 'xtbopt.xyz', chrg, os.path.join("/".join(conf_path.split("/")[:-2]),
        # 'full_opt', conf_name+'_full_opt'),
        # ncores=num_cpu_single, mem=(int(mem_gb)/2)*1000)
        final_conf_energies.append(conf_energy)
        final_conf_mols.append(conf_mol)

    # Get only the lowest energy conformer
    minE_index = np.argmin(final_conf_energies)
    best_conf_mol = final_conf_mols[minE_index]
    best_conf_energy_xtb = final_conf_energies[minE_index]

    # uncomment when doing single point calculations on all unique conformers
    if best_conf_energy_xtb != float(99999):
        # runs a Orca single point calculation on the lowest xTB energy conformer
        best_conf_energy_sp = run_orca(
            xyz_file="xtbopt.xyz",
            chrg=chrg,
            path=conf_paths[minE_index],
            ncores=num_cpu_single / 2,
            mem=(int(mem_gb)) * 1000,
            functional=functional,
            basis_set=basis,
            dispersion=dispersion,
            optfreq=optfreq,
            solvent_name=solvent_name,
        )

        if best_conf_energy_sp == float("inf"):
            best_conf_energy_sp = rerun_orca(
                chrg=chrg,
                path=conf_paths[minE_index],
                ncores=num_cpu_single / 2,
                mem=(int(mem_gb)) * 1000,
                functional=functional,
                basis_set=basis,
                dispersion=dispersion,
                optfreq=optfreq,
                solvent_name=solvent_name,
            )
    else:
        best_conf_energy_sp = float("inf")

    # START - CLEAN UP #
    for conf_name in conf_names_copy:
        # 'calc' sys.argv[2]
        conf_path = os.path.join(
            os.getcwd().replace("/src/pKalculator", ""),
            calc_path,
            conf_name.split("_")[0],
            conf_name.split("_")[1],
        )

        if os.path.isfile(os.path.join(conf_path, conf_name + "_gfnff.xyz")):
            os.remove(os.path.join(conf_path, conf_name + "_gfnff.xyz"))

        if os.path.isfile(
            os.path.join(
                conf_path, conf_name + "_gfn" + method.replace(" ", "") + ".xyz"
            )
        ):
            os.remove(
                os.path.join(
                    conf_path, conf_name + "_gfn" + method.replace(" ", "") + ".xyz"
                )
            )

        # Remove GFNFF-xTB folder
        folder_path = os.path.join(conf_path, "gfnff")
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if os.path.isfile(f"{folder_path}/{file_remove}"):
                    os.remove(f"{folder_path}/{file_remove}")
            # checking whether the folder is empty or not
            if len(os.listdir(folder_path)) == 0:
                os.rmdir(folder_path)
            else:
                print("Folder is not empty")
        # --------------------------
        # CLEAN GFN?-xTB folder
        # --------------------------
        folder_path = os.path.join(conf_path, "gfn" + method.replace(" ", ""))
        # 'xtbopt.log' removed from file_remove_list
        # file_remove_list = [
        #     "charges",
        #     "coordprot.0",
        #     "lmocent.coord",
        #     "orca_calc_atom46.densities",
        #     "orca_calc_atom46.out",
        #     "orca_calc_atom46_property.txt",
        #     "orca_calc_atom53.densities",
        #     "orca_calc_atom53.out",
        #     "orca_calc_atom53_property.txt",
        #     "orca_calc.cpcm",
        #     "orca_calc.densities",
        #     "orca_calc.gbw",
        #     "wbo",
        #     "xtblmoinfo",
        #     ".xtboptok",
        #     "xtbrestart",
        #     "xtbscreen.xyz",
        # ]

        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if (
                    file_remove.split(".")[-1]
                    in ["sdf", "xtbout", "xyz", "inp", "out", "log"]
                    and file_remove != "xtbscreen.xyz"
                ):
                    continue
                elif os.path.isfile(f"{folder_path}/{file_remove}"):
                    os.remove(f"{folder_path}/{file_remove}")
        # --------------------------
        # END - CLEAN UP
        # --------------------------
    return best_conf_energy_xtb, best_conf_energy_sp, best_conf_mol


def calculate_relative_energy(mol, name, smi, neutral_energy_xtb, neutral_energy_sp):
    try:
        (
            best_conf_energy_deprot_xtb,
            best_conf_energy_deprot_sp,
            best_conf_mol_deprot,
        ) = calculateEnergy(rdkit_mol=mol, name=name, smi=smi)

        E_rel_xtb = (
            float("inf")
            if neutral_energy_xtb == float("inf")
            else best_conf_energy_deprot_xtb - neutral_energy_xtb
        )
        E_rel_sp = (
            float("inf")
            if neutral_energy_sp == float("inf")
            else best_conf_energy_deprot_sp - neutral_energy_sp
        )

        return (
            best_conf_energy_deprot_xtb,
            best_conf_energy_deprot_sp,
            E_rel_xtb,
            E_rel_sp,
            best_conf_mol_deprot,
        )

    except Exception as e:
        print(f"Error during energy calculation for {name} {smi}")
        print(e)
        return float("inf"), float("inf"), float("inf"), float("inf")


def get_relative_energy(
    input_smiles: str, name: str, rxn_type: str = "rm_proton"
) -> tuple:
    lst_best_conf_mol_deprot = []
    lst_e_deprot_xtb = []
    lst_e_deprot_sp = []
    lst_e_rel_deprot_xtb = []
    lst_e_rel_deprot_sp = []

    mol_neutral = Chem.MolFromSmiles(input_smiles)
    (
        best_conf_energy_neutral_xtb,
        best_conf_energy_neutral_sp,
        best_conf_mol,
    ) = calculateEnergy(rdkit_mol=mol_neutral, name=name, smi=input_smiles)

    (
        ref_mol_smiles_map,
        lst_name_deprot,
        lst_smiles_deprot_map,
        lst_smiles_deprot,
        lst_atommapnum,
    ) = deprotonate(name=name, smiles=input_smiles, rxn=rxn_type)
    lst_mol_deprot = [Chem.MolFromSmiles(smi) for smi in lst_smiles_deprot]

    for name_deprot, smi_deprot, mol_deprot in list(
        zip(lst_name_deprot, lst_smiles_deprot, lst_mol_deprot)
    ):
        print(f"starting calculation for {name_deprot} {smi_deprot}")
        (
            best_conf_energy_deprot_xtb,
            best_conf_energy_deprot_sp,
            E_rel_xtb,
            E_rel_sp,
            best_conf_mol_deprot,
        ) = calculate_relative_energy(
            mol_deprot,
            name_deprot,
            smi_deprot,
            best_conf_energy_neutral_xtb,
            best_conf_energy_neutral_sp,
        )

        lst_best_conf_mol_deprot.append(best_conf_mol_deprot)
        lst_e_deprot_xtb.append(best_conf_energy_deprot_xtb)
        lst_e_deprot_sp.append(best_conf_energy_deprot_sp)
        lst_e_rel_deprot_xtb.append(E_rel_xtb)
        lst_e_rel_deprot_sp.append(E_rel_sp)

    return (
        ref_mol_smiles_map,
        lst_name_deprot,
        lst_smiles_deprot,
        lst_smiles_deprot_map,
        lst_atommapnum,
        best_conf_mol,
        lst_best_conf_mol_deprot,
        best_conf_energy_neutral_xtb,
        best_conf_energy_neutral_sp,
        lst_e_deprot_xtb,
        lst_e_deprot_sp,
        lst_e_rel_deprot_xtb,
        lst_e_rel_deprot_sp,
    )


def control_calcs(df, method, solvent_model, solvent_name):
    # Initialize new columns
    new_columns = [
        "ref_mol_smiles_map",
        "lst_name_deprot",
        "lst_smiles_deprot_map",
        "lst_smiles_deprot",
        "lst_atomsite",
        "mol_neutral",
        "lst_mol_deprot",
        "e_neutral_xtb",
        "e_neutral_sp",
        "lst_e_xtb",
        "lst_e_rel_xtb",
        "lst_e_sp",
        "lst_e_rel_sp",
        "gfn_method",
        "solvent_model",
        "solvent_name",
    ]

    for col in new_columns:
        df[col] = None

    # Store values in a dictionary
    results = {}

    # ensures that leading or trailing whitespace is removed from column names
    df.columns = df.columns.str.strip()

    for idx, row in df.iterrows():
        name = row["names"]
        smiles = row["smiles"]

        try:
            (
                ref_mol_smiles_map,
                lst_name_deprot,
                lst_smiles_deprot,
                lst_smiles_deprot_map,
                lst_atommapnum,
                best_conf_mol,
                lst_best_conf_mol_deprot,
                best_conf_energy_neutral_xtb,
                best_conf_energy_neutral_sp,
                lst_e_deprot_xtb,
                lst_e_deprot_sp,
                lst_e_rel_deprot_xtb,
                lst_e_rel_deprot_sp,
            ) = get_relative_energy(smiles, name=name, rxn_type="rm_proton")

            values = {
                "ref_mol_smiles_map": ref_mol_smiles_map,
                "lst_name_deprot": lst_name_deprot,
                "lst_smiles_deprot_map": lst_smiles_deprot_map,
                "lst_smiles_deprot": lst_smiles_deprot,
                "lst_atomsite": lst_atommapnum,
                "mol_neutral": best_conf_mol,
                "lst_mol_deprot": lst_best_conf_mol_deprot,
                "e_neutral_xtb": best_conf_energy_neutral_xtb,
                "e_neutral_sp": best_conf_energy_neutral_sp,
                "lst_e_xtb": lst_e_deprot_xtb,
                "lst_e_rel_xtb": lst_e_rel_deprot_xtb,
                "lst_e_sp": lst_e_deprot_sp,
                "lst_e_rel_sp": lst_e_rel_deprot_sp,
                "gfn_method": f"gfn{str(method)}",
                "solvent_model": solvent_model,
                "solvent_name": solvent_name,
            }

            results[idx] = values

        except Exception as e:
            print(f"WARNING! pKalculator failed for {name} {smiles}")
            print(e)

    # Update dataframe outside the loop
    for idx, values in results.items():
        for col, value in values.items():
            df.at[idx, col] = value

    return df


if __name__ == "__main__":
    import pandas as pd
    import submitit

    args = get_args()

    method = 2
    solvent_model = "alpb"
    csv_path = args.csv_path
    calc_path = args.calc_path
    functional = args.functional
    basis = args.basis
    solvent_name = args.solvent
    optfreq = args.optfreq
    dispersion = args.dispersion

    path_data = Path(Path.home() / "pKalculator" / "data" / "external")

    # #Bordwell dataset https://organicchemistrydata.org/hansreich/resources/pka/
    # "bordwell" / "bordwellCH.exp"

    # bordwell no CH dataset https://organicchemistrydata.org/hansreich/resources/pka/
    # bordwell/bordwellnoCH.exp

    # # shen_et_al dataset
    # "shen_et_al/shen_et_al.smiles"

    # ibond dataset
    # "ibond"/"ibond_data.smiles"

    # dataset from https://github.com/C-H-activation/ICB-validation/blob/main/smiles
    # "icb"/"ICB_validation.smiles"

    df = pd.read_csv(path_data / csv_path, sep=",")

    executor = submitit.AutoExecutor(
        folder=args.submitit_path
    )  # "submitit_pKalculator_old" sys.argv[2]
    executor.update_parameters(
        name="pKalc",  # pKalculator
        cpus_per_task=int(num_cpu_single),
        mem_gb=int(mem_gb),
        timeout_min=60000,  # 500 hours -> 20 days : 60000 --> 1000 hours -> 41 days
        slurm_partition="kemi1",
        slurm_array_parallelism=50,
    )
    print(executor)

    jobs = []
    with executor.batch():
        chunk_size = 1
        for start in range(0, df.shape[0], chunk_size):
            df_subset = df.iloc[start : start + chunk_size]
            job = executor.submit(
                control_calcs, df_subset, method, solvent_model, solvent_name
            )
            jobs.append(job)
