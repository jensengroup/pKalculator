# utility functions to Extract, Transform, and Load data

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import ast
from rdkit import Chem
from rdkit.Chem import Draw
import re
import sys

sys.path.insert(0, "../smi2gcs")
# FIX THIS WHEN WORKING IN ANOTHER DIRECTORY
# from DescriptorCreator.PrepAndCalcDescriptor import Generator


path_data = Path(Path.home() / "pKalculator" / "data" / "external")
path_submitit = Path(
    Path.home() / "pKalculator" / "data" / "raw" / "bordwell" / "submitit"
)


def load_pickles(folder):
    # print(folder)
    return pd.concat(
        [
            pd.read_pickle(pkl)[1].assign(file=pkl.name)
            for pkl in folder.glob("*_result.pkl")
        ],
        ignore_index=True,
    )  # .sort_values(by='file')


def find_group(smiles, group_smarts):
    try:
        smarts = Chem.MolFromSmarts(group_smarts)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        return rdkit_mol.HasSubstructMatch(smarts)
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_group_pattern(smiles):
    """Smarts patterns are based on the Grybowski paper with an addition of aliphatic and allylic"""
    df_group_patterns = pd.read_csv(
        path_data / "grzybowski/group_pattern_grzybowski2.smarts", delimiter=" "
    )
    lst_match_idx = []
    t_mol = Chem.MolFromSmiles(smiles)
    lst_match_idx = [
        idx
        for idx, row_group_pattern in enumerate(
            df_group_patterns.itertuples(index=False)
        )
        if t_mol.HasSubstructMatch(Chem.MolFromSmarts(row_group_pattern.group_pattern))
    ]
    if len(lst_match_idx) < 1:
        return ("unknown", "unknown")
    else:
        best_match_idx = min(lst_match_idx)
        return (
            df_group_patterns.at[best_match_idx, "group_name"],
            df_group_patterns.at[best_match_idx, "group_pattern"],
        )


def get_no_convergence(source_dir, folder):
    """
    iterate over all calculation folders and find the ones that did not converge and the ones that terminated with an error

    example:
    soruce_dir = 'bordwell'
    folder = 'calc_optfreq_r2SCAN-3c-bordwellCH'
    p.parent.parent.parent.name --> compXXX
    p.parent.parent.name --> confXX
    """
    lst_no_convergence = []
    lst_termination = []
    pattern = re.compile(r"[a-z]+([0-9]+)", re.I)

    orca_out_paths = list(
        (
            Path.home() / "pKalculator" / "data" / "raw" / source_dir / "calc" / folder
        ).rglob("orca_calc*.out")
    )

    # Create a dictionary to group files by parent directory
    file_groups = {}
    for orca_out_path in orca_out_paths:
        parent_dir = orca_out_path.parent
        if parent_dir not in file_groups:
            file_groups[parent_dir] = []
        file_groups[parent_dir].append(orca_out_path)

    # Iterate through grouped files and filter as needed
    for parent_dir, files in file_groups.items():
        if "orca_calc.out" in [
            file.name for file in files
        ] and "orca_calc_rerun.out" in [file.name for file in files]:
            # Only keep "orca_calc_rerun.out" if both files exist in the same directory
            files = [file for file in files if file.name == "orca_calc_rerun.out"]
        for file in files:
            with open(file, "r") as f:
                for i, line in enumerate(f):
                    if (
                        "The optimization did not converge but reached the maximum"
                        in line
                    ):
                        match = pattern.match(parent_dir.parent.parent.name)
                        if match:
                            lst_no_convergence.append(
                                (
                                    match.group(1),
                                    parent_dir.parent.parent.name,
                                    parent_dir.parent.name,
                                )
                            )
                    elif "ORCA finished by error termination" in line:
                        match = pattern.match(parent_dir.parent.parent.name)
                        if match:
                            lst_line = line.strip().split(" ")
                            last_part = (
                                lst_line[-1]
                                if "in" in lst_line[-2]
                                else " ".join(lst_line[-2:])
                            )
                            lst_termination.append(
                                (
                                    match.group(1),
                                    parent_dir.parent.parent.name,
                                    parent_dir.parent.name,
                                    last_part,
                                )
                            )

    lst_no_convergence.sort(key=lambda x: int(x[0]))
    lst_termination.sort(key=lambda x: int(x[0]))

    return lst_no_convergence, lst_termination


def correct_conv_term(df, lst_no_convergence, lst_termination):
    lst_err = [x[1] for x in lst_termination] + [x[1] for x in lst_no_convergence]
    # Create 'lst_idx_err_term_conv' column with indices of 'lst_name_deprot' elements that match 'lst_err'

    df["lst_idx_err_term_conv"] = df["lst_name_deprot"].apply(
        lambda x: [i for i, name_deprot in enumerate(x) if name_deprot in lst_err]
        if isinstance(x, list)
        else [np.nan]
    )

    # Iterate over rows and update 'lst_e_rel_sp' based on 'lst_idx_err_term_conv'
    df["lst_e_rel_sp"] = df.apply(
        lambda row: [
            float("inf") if i in row["lst_idx_err_term_conv"] else energy
            for i, energy in enumerate(row["lst_e_rel_sp"])
            if isinstance(row["lst_e_sp"], list)
        ]
        if isinstance(row["lst_e_rel_sp"], list)
        else row["lst_e_rel_sp"],
        axis=1,
    )

    return df


def prepare_dataframe(df):
    """
    This function prepares the dataframe for further analysis.
    It removes empty lists, removes amines and alcohols, and processes the dataframe by adding extra columns.

    """
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    # remove column that is troubled

    # iterate over df and create mol objects for each compound. Check if compound is an amine or alcohol and add True/False to df
    amines = "[NX3;H2,H1;!$(NC=O)]"
    alcohol = "[#6][OX2H]"
    amides = "[NX3;H2,H1][CX3](=[OX1])[#6]"  # For standard amide smarts: [NX3;H2,H1][CX3](=[OX1])[#6]
    df["amine"] = df["smiles"].apply(find_group, group_smarts=amines)
    df["alcohol"] = df["smiles"].apply(find_group, group_smarts=alcohol)
    df["amide"] = df["smiles"].apply(find_group, group_smarts=amides)

    # get group pattern from grzybowski
    df["group_pattern"] = df["smiles"].apply(get_group_pattern)

    df.reset_index(drop=True, inplace=True)

    return df


def find_diff_hyb(df):
    """
    This function finds the difference in hybridization between the neutral and deprotonated molecule.
    Returns a tuple of:
    - the idx of the dataframe
    - idx for the deprotonated smile where hyb is different from the neutral molecule in the list of smiles
    - atom map number
    """
    # find difference in hybridization
    lst_diff_hybrid = []
    lst_same_hybrid = []
    for idx, row in df.iterrows():
        # print(idx)
        react_mol_neutral = Chem.MolFromSmiles(row["ref_mol_smiles_map"])
        for deprot_idx, deprot_smi in enumerate(row["lst_smiles_deprot_map"]):
            deprot_mol = Chem.MolFromSmiles(deprot_smi)
            for atom_idx, atom in enumerate(deprot_mol.GetAtoms()):
                charge = atom.GetFormalCharge()
                atom_mapnum = atom.GetAtomMapNum()
                if charge == -1 and atom.GetSymbol() == "C":
                    for a_idx, a in enumerate(react_mol_neutral.GetAtoms()):
                        # if the hybridization between the neutral molecule and the deprotonated molecule is not the same
                        if (
                            a.GetAtomMapNum() == atom_mapnum
                            and a.GetSymbol() == "C"
                            and a.GetHybridization() != atom.GetHybridization()
                        ):
                            lst_diff_hybrid.append((idx, deprot_idx, atom_mapnum))
                        if (
                            a.GetAtomMapNum() == atom_mapnum
                            and a.GetSymbol() == "C"
                            and a.GetHybridization() == atom.GetHybridization()
                        ):
                            lst_same_hybrid.append((idx, deprot_idx, atom_mapnum))
    return lst_diff_hybrid, lst_same_hybrid


def get_cm5_desc_vector(smi_name, smi_map, lst_atom_index, n_shells=6):
    generator = Generator()
    des = (
        "GraphChargeShell",
        {"charge_type": "cm5", "n_shells": n_shells, "use_cip_sort": True},
    )
    try:
        cm5_list = generator.calc_CM5_charges(
            smi_map, name=smi_name, optimize=False, save_output=True
        )
        atom_indices, descriptor_vector = generator.create_descriptor_vector(
            lst_atom_index, des[0], **des[1]
        )
    except Exception:
        descriptor_vector = None

    return cm5_list, atom_indices, descriptor_vector


# ----------------------------------#
# SDF files
# ----------------------------------#


def write_sdf(folder=""):
    """
    Writes an sdf file with the information from the *_opt.sdf file
    Works for xtb output files.
    Need testing for orca files.
    """

    folder_path = Path(
        Path.home() / "pKalculator" / "data" / "raw" / "bordwell" / "calc" / folder
    )
    pattern = re.compile(r"[a-z]+([0-9]+)", re.I)

    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.name == "orca_calc.out":
            # skips deprotonated files
            if "#" in file_path.parent.parent.parent.name:
                continue
            else:
                for sdf in Path(file_path.parent).glob("*_opt.sdf"):
                    if sdf.is_file():
                        sdf_file_path = sdf
                        suppl = Chem.SDMolSupplier(
                            str(sdf_file_path),
                            sanitize=False,
                            removeHs=False,
                            strictParsing=True,
                        )
                        w = Chem.SDWriter(
                            str(file_path.parent)
                            + f"/{sdf_file_path.name.split('.')[0]}_info.sdf"
                        )
                        match = pattern.match(file_path.parent.parent.parent.name)
                        for mol in suppl:
                            # print(mol.GetProp('_Name'))
                            # name = mol.GetProp('_Name').split(' ')[-1].split('/')[-4]
                            mol.SetProp("_Name", file_path.parent.parent.parent.name)
                            if match:
                                mol.SetProp("ID", match.group(1))
                            mol.SetProp("Mol_ID", file_path.parent.parent.parent.name)
                            mol.SetProp("Smiles", Chem.MolToSmiles(mol))
                            w.write(mol)
                        w.close()
    return


def merge_sdf(input_folder="", output_file=""):
    folder_path = Path(
        Path.home()
        / "pKalculator"
        / "data"
        / "raw"
        / "bordwell"
        / "calc"
        / input_folder
    )
    file_path_output = Path(
        Path.home() / "pKalculator" / "data" / "processed" / "bordwell" / output_file
    )
    # folder_path = Path('/groups/kemi/borup/pKalculator/data/raw/bordwell/calc/calc_sp_CAM-B3LYP-D4_def2-TZVPPD-BordwellCH')
    # Path('/groups/kemi/borup/pKalculator/data/raw/bordwell/calc/calc_sp_CAM-B3LYP-D4_def2-TZVPPD-BordwellCH')
    sdfs = []
    pattern = re.compile(r"[a-z]+([0-9]+)", re.I)
    # Iterate over files recursively in the folder
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.name == "orca_calc.out":
            # skips deprotonated files
            if "#" in file_path.parent.parent.parent.name:
                continue
            else:
                # Get the corresponding .sdf file
                for sdf in Path(file_path.parent).glob("*_opt_info.sdf"):
                    if sdf.is_file():
                        match = pattern.match(file_path.parent.parent.parent.name)
                        if match:
                            sdf_file_path = sdf
                            sdfs.append((match.group(1), sdf_file_path))

    sdfs.sort(key=lambda x: int(x[0]))

    # Open the output file in append mode
    with open(file_path_output, "a") as out_file:
        # Get the corresponding .sdf file
        for sdf in (x[1] for x in sdfs):
            if sdf.is_file():
                # Process the .sdf file
                with open(sdf, "r") as sdf_file:
                    sdf_content = sdf_file.read()
                    # Write the content to the output file
                    out_file.write(sdf_content)
    return


def calc_pka_lfer(e_rel: float) -> float:
    pka = 0.5941281 * e_rel - 159.33107321
    return pka


def pka_dmso_to_pka_thf(pka: float, reverse=False) -> float:
    pka_thf = -0.963 + 1.046 * pka
    if reverse:
        pka_dmso = (pka_thf + 0.963) / 1.046
        return pka_dmso

    return pka_thf


if __name__ == "__main__":
    path_submitit = Path(
        Path.home() / "pKalculator" / "data" / "raw" / "bordwell" / "submitit"
    )
    path_calc = Path(Path.home() / "pKalculator" / "data" / "raw" / "bordwell" / "calc")
    folder = "submitit_optfreq_CAM-B3LYP-D4_bordwellCH"
    path_folder = Path(path_submitit / folder)
    print(f"searching for pickles in : {path_folder}")
    print("--" * 20, "PICKLES found", "--" * 20)
    print("--" * 20, folder, "--" * 20)

    df_results = load_pickles(folder=path_folder)
    df_results = prepare_dataframe(df=df_results)

    print("checks for convergence and termination issues")
    lst_no_convergence, lst_termination = get_no_convergence(
        source_dir="bordwell", folder="calc_optfreq_CAM-B3LYP-D4_bordwellCH"
    )
    print("Correcting for convergence and termination issues")
    df_results = correct_conv_term(df_results, lst_no_convergence, lst_termination)

    df_results = df_results[~df_results.ref_mol_smiles_map.isna()]
    df_results.reset_index(drop=True, inplace=True)

    # find difference in hybridization
    # lst_diff_hybrid, lst_same_hybrid = find_diff_hyb(df_results)
    # df_results["lst_diff_hybrid"] = df_results.apply(
    #     lambda row: [
    #         (deprot_idx, atom_mapnum)
    #         for idx, deprot_idx, atom_mapnum in lst_diff_hybrid
    #         if idx == row.name
    #     ],
    #     axis=1,
    # )
    # df_results["lst_same_hybrid"] = df_results.apply(
    #     lambda row: [
    #         (deprot_idx, atom_mapnum)
    #         for idx, deprot_idx, atom_mapnum in lst_same_hybrid
    #         if idx == row.name
    #     ],
    #     axis=1,
    # )

    df_results["atom_index"] = df_results.lst_atomsite.apply(
        lambda x: [i - 1 for i in x]
    )
    df_results["e_rel_min"] = df_results.apply(
        lambda row: min(row["lst_e_rel_sp"]), axis=1
    )
    df_results["lst_pka_lfer"] = df_results.lst_e_rel_sp.apply(
        lambda x: [calc_pka_lfer(e_rel) for e_rel in x]
    )
    df_results["pka_min_sp"] = df_results.lst_pka_lfer.apply(lambda x: min(x))
    # set atom index to 0 if it is not the index with the minimum pka value
    # df_results['atom_lowest'] = df_results.apply(lambda row : [0 if i != row.pka_min_sp else 1 for i in row.lst_pka_sp], axis=1)
    print("calculating cm5 charges and descriptor vectors")
    output = df_results.apply(
        lambda row: get_cm5_desc_vector(
            smi_name=row["name"],
            smi_map=row["ref_mol_smiles_map"],
            lst_atom_index=row["atom_index"],
            n_shells=6,
        ),
        axis=1,
    )
    (
        df_results["cm5"],
        df_results["atom_indices"],
        df_results["descriptor_vector"],
    ) = zip(*output)

    print("saving results to pickle")
    # save df_results
    df_results.to_pickle(
        Path(Path.home() / "pKalculator" / "data" / "test_result_20231123.pkl")
    )
