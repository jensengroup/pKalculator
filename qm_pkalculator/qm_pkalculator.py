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
import sys
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
from run_orca import run_orca, rerun_orca
from modify_smiles import deprotonate, remove_Hs

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

home_directory = Path.cwd()
if home_directory.name != "pKalculator":
    raise ValueError("Please run this script from the pKalculator directory")
sys.path.append(str(home_directory / "qm_pkalculator"))
os.chdir(home_directory / "qm_pkalculator")

method = 2
solvent_model = "alpb"
num_cpu_single = None
mem_gb = None
csv_path = None
calc_path = None
functional = None
basis = None
solvent_name = None
opt = None
freq = None
dispersion = None


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Get deprotonated energy values of a molecule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-cpus",
        "--cpus",
        metavar="cpus",
        help="Number of cpus per job",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-mem",
        "--mem",
        metavar="mem",
        help="Memory in GB per job",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-p",
        "--partition",
        metavar="partition",
        help="Enter the partition to use for the calculations. Default is kemi1",
        default="kemi1",
        type=str,
    )

    parser.add_argument(
        "-csv",
        "--csv_path",
        metavar="csv_path",
        help="csv path. The csv file must be comma seperated and contain a column named 'smiles' and a column named 'names'",
        default="data/qm_calculations/test.csv",
        type=str,
    )

    parser.add_argument(
        "-calc",
        "--calc_path",
        metavar="calc_path",
        help="calculation path where the calculations are stored.",
        default="data/qm_calculations/calc_test",
        type=str,
    )

    parser.add_argument(
        "-submit",
        "--submit_path",
        metavar="submit_path",
        help="submit path where log files are stored.",
        default="data/qm_calculations/submit_test",
        type=str,
    )

    parser.add_argument(
        "-f",
        "--functional",
        metavar="functional",
        help="Add the functional to use for ORCA calculations. Default is CAM-B3LYP.",
        default="CAM-B3LYP",
        type=str,
    )

    parser.add_argument(
        "-b",
        "--basis",
        metavar="basis",
        help="Enter the basis set to use for ORCA calculations. Default is def2-TZVPPD.",
        default="def2-TZVPPD",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--solvent",
        metavar="solvent",
        help="Enter solvent name. Default is DMSO.",
        default="DMSO",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dispersion",
        help="set dispersion to use for ORCA calculations. Default is False.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--opt",
        help="set optimization to use for ORCA calculations. Default is False.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--freq",
        help="set frequency to use for ORCA calculations. Default is False.",
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
    global num_cpu_single, calc_path

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


def get_multiple_confs(name, smi, rdkit_mol, n_conformers):
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
                ps.maxAttempts = 1000
                cids = AllChem.EmbedMultipleConfs(
                    rdkit_mol, numConfs=n_conformers * 5, params=ps
                )
                if not cids:
                    raise Exception(f"4rd embed failed for {name} with SMILES: {smi}")

    # Unpack confomers and assign conformer names
    conf_mols = [
        Chem.Mol(rdkit_mol, False, i) for i in range(rdkit_mol.GetNumConformers())
    ]

    conf_names = [
        name + f"_conf{str(i+1).zfill(2)}" for i in range(rdkit_mol.GetNumConformers())
    ]

    return conf_mols, conf_names


def check_bridgehead(name, smiles, smiles_neutral, atomsite):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    query_bridgehead_mol = Chem.MolFromSmarts("[Ax{3-4}]")  # '[x{3-4}]']

    lst_conf_mols_deprot = []
    lst_conf_names_deprot = []
    if rdkit_mol.GetSubstructMatches(query_bridgehead_mol):
        # use the neutral smiles to generate conformers
        rdkit_mol = Chem.MolFromSmiles(smiles_neutral)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        rot_bond = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdkit_mol)
        n_conformers = min(10 + 3 * rot_bond, 20)
        lst_conf_mols_neutral, lst_conf_names_deprot_prelim = get_multiple_confs(
            name=name, smi=smiles, rdkit_mol=rdkit_mol, n_conformers=n_conformers
        )

        for conf_mol_neutral, conf_name_deprot in zip(
            lst_conf_mols_neutral, lst_conf_names_deprot_prelim
        ):
            (
                lst_atom_index_deprot,
                lst_atomsite_deprot,
                lst_smiles_deprot,
                lst_smiles_map_deprot,
                lst_mol_deprot,
                lst_names_deprot,
                dict_atom_idx_to_H_indices,
            ) = remove_Hs(
                name=name,
                smiles=None,
                rdkit_mol=conf_mol_neutral,
                atomsite=atomsite,
                gen_all=False,
                remove_H=False,
            )
            lst_conf_mols_deprot.append(
                lst_mol_deprot[0]
            )  # only take the first as the list is 1 long
            lst_conf_names_deprot.append(conf_name_deprot)

        return lst_conf_mols_deprot, lst_conf_names_deprot


def calculateEnergy(
    rdkit_mol: Chem.rdchem.Mol,
    name: str,
    smi: str,
    bridgehead_deprot: bool,
    conf_names: list,
    conf_mols: list,
):
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

    global calc_path, num_cpu_single
    global method, solvent_model, solvent_name
    global functional, basis, dispersion, opt, freq

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
    query_bridgehead_mol = Chem.MolFromSmarts("[Ax{3-4}]")  # '[x{3-4}]']
    if rdkit_mol.GetSubstructMatches(query_bridgehead_mol):
        n_conformers = min(10 + 3 * rot_bond, 20)

    if not bridgehead_deprot:
        conf_mols, conf_names = get_multiple_confs(
            name=name, smi=smi, rdkit_mol=rdkit_mol, n_conformers=n_conformers
        )

    conf_names_copy = copy.deepcopy(conf_names)

    # Run a GFN-FF optimization
    print("--running GFN-FF--")
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
    print("--running GFN2-xTB--")
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
        # runs a Orca on the lowest xTB energy conformer
        print("--running Orca--")
        best_conf_energy_sp = run_orca(
            xyz_file="xtbopt.xyz",
            chrg=chrg,
            path=conf_paths[minE_index],
            ncores=num_cpu_single / 2,
            mem=(int(mem_gb)) * 1000,
            functional=functional,
            basis_set=basis,
            dispersion=dispersion,
            opt=opt,
            freq=freq,
            solvent_name=solvent_name,
        )

        if best_conf_energy_sp == float("inf"):
            print("--restarting Orca--")
            best_conf_energy_sp = rerun_orca(
                chrg=chrg,
                path=conf_paths[minE_index],
                ncores=num_cpu_single / 2,
                mem=(int(mem_gb)) * 1000,
                functional=functional,
                basis_set=basis,
                dispersion=dispersion,
                opt=opt,
                freq=freq,
                solvent_name=solvent_name,
            )
    else:
        best_conf_energy_sp = float("inf")

    # START - CLEAN UP #
    for conf_name in conf_names_copy:
        # 'calc' sys.argv[2]
        conf_path = os.path.join(
            os.getcwd().replace("/pKalculator", ""),
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


def control_calcs(df, method, solvent_model, solvent_name):
    # Initialize new columns
    new_columns = [
        "mol",
        "e_xtb",
        "e_dft",
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
        smiles_neutral = row["smiles_neutral"]
        atomsite = row["atomsite"]
        print(f"Running QM pKalculator for {name} {smiles}")
        try:
            mol = Chem.MolFromSmiles(smiles)
            query_bridgehead_mol = Chem.MolFromSmarts("[Ax{3-4}]")
            if (
                mol.GetSubstructMatches(query_bridgehead_mol)
                and smiles != smiles_neutral
                and atomsite != -1
            ):
                bridgehead_deprot = True
                print(f"bridgehead_deprot: {bridgehead_deprot}")

                lst_conf_mols_deprot, lst_conf_names_deprot = check_bridgehead(
                    name=name,
                    smiles=smiles,
                    smiles_neutral=smiles_neutral,
                    atomsite=atomsite,
                )

                (
                    best_conf_energy_neutral_xtb,
                    best_conf_energy_neutral_sp,
                    best_conf_mol,
                ) = calculateEnergy(
                    rdkit_mol=mol,
                    name=name,
                    smi=smiles,
                    bridgehead_deprot=bridgehead_deprot,
                    conf_names=lst_conf_names_deprot,
                    conf_mols=lst_conf_mols_deprot,
                )

            else:
                bridgehead_deprot = False
                print(f"bridgehead_deprot: {bridgehead_deprot}")
                (
                    best_conf_energy_neutral_xtb,
                    best_conf_energy_neutral_sp,
                    best_conf_mol,
                ) = calculateEnergy(
                    rdkit_mol=mol,
                    name=name,
                    smi=smiles,
                    bridgehead_deprot=bridgehead_deprot,
                    conf_names=None,
                    conf_mols=None,
                )
            # "mol": best_conf_mol,
            values = {
                "e_xtb": best_conf_energy_neutral_xtb,
                "e_dft": best_conf_energy_neutral_sp,
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


def gen_deprot_df(df):
    lst_deprot_df = []
    for idx, row in df.iterrows():
        (
            ref_mol_smiles_map,
            lst_name_deprot,
            lst_smiles_deprot_map,
            lst_smiles_deprot,
            lst_atommapnum,
            lst_atomidx,
        ) = deprotonate(name=row["names"], smiles=row["smiles"], rxn="rm_proton")

        df_neg = pd.DataFrame(
            {
                "names": lst_name_deprot,
                "smiles": lst_smiles_deprot,
                "smiles_deprot_map": lst_smiles_deprot_map,
                "atomsite": lst_atommapnum,
                "atom_index": lst_atomidx,
                "ref_name": row["names"],
                "ref_mol_smiles_map": ref_mol_smiles_map,
                "neutral": False,
                "smiles_neutral": row["smiles"],
            }
        )
        lst_deprot_df.append(df_neg)

    df_neg = pd.concat(lst_deprot_df, ignore_index=True)
    return df_neg


if __name__ == "__main__":
    import pandas as pd
    import submitit
    from datetime import datetime

    # example usage:
    # python qm_pkalculator/qm_pkalculator.py -cpus 10 -mem 20 -calc data/qm_calculations/calc_test -submit data/qm_calculations/submit_test -f CAM-B3LYP -b  def2-TZVPPD -d -o -q
    args = get_args()

    method = 2
    solvent_model = "alpb"

    num_cpu_single = args.cpus
    mem_gb = args.mem
    partition = args.partition
    csv_path = args.csv_path
    calc_path = args.calc_path
    submit_path = args.submit_path
    functional = args.functional
    basis = args.basis
    solvent_name = args.solvent
    opt = args.opt
    freq = args.freq
    dispersion = args.dispersion

    path_data = Path(home_directory / "data" / "qm_calculations")
    calc_path = home_directory / calc_path
    submit_path = home_directory / submit_path
    csv_path = home_directory / csv_path
    prelim_name = Path(calc_path).name

    print(f"calc_path: {calc_path}")
    print(f"submit_path: {submit_path}")
    print(f"csv_path: {csv_path}")

    df = pd.read_csv(csv_path, sep=",")
    df["ref_name"] = df["names"]
    df["neutral"] = True

    df_neg = gen_deprot_df(df=df)
    df_merged = pd.concat([df, df_neg])
    df_merged.sort_values(by=["ref_name", "names"], inplace=True)
    # Resetting the index
    df_merged.reset_index(drop=True, inplace=True)
    df_merged["atomsite"] = df_merged["atomsite"].fillna(-1).astype(int)
    df_merged["atomsite"] = df_merged["atomsite"].astype(int)
    df_merged["atom_index"] = df_merged["atom_index"].fillna(-1).astype(int)
    df_merged["atom_index"] = df_merged["atom_index"].astype(int)

    df_merged.to_pickle(
        path_data / f"df_prelim_{prelim_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
    )
    print("saving preliminary dataframe to:")
    print(
        path_data / f"df_prelim_{prelim_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
    )

    executor = submitit.AutoExecutor(folder=submit_path)
    executor.update_parameters(
        name="pKalc",  # pKalculator
        cpus_per_task=int(num_cpu_single),
        mem_gb=int(mem_gb),
        timeout_min=500,  # 500 hours -> 20 days
        slurm_partition=f"{partition}",
        slurm_array_parallelism=50,
    )
    print(executor)

    jobs = []
    with executor.batch():
        chunk_size = 1
        for start in range(0, df_merged.shape[0], chunk_size):
            df_subset = df_merged.iloc[start : start + chunk_size]
            job = executor.submit(
                control_calcs, df_subset, method, solvent_model, solvent_name
            )
            jobs.append(job)
