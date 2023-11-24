import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from ase.units import Hartree, mol, kcal
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def get_xyz_to_mol(file, charge=0):
    """

    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(Path(file), "r") as file:
        ind = file.read()
    try:
        raw_mol = Chem.MolFromXYZBlock(ind)
        conn_mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        rdDetermineBonds.DetermineBondOrders(conn_mol, charge=charge)
        cm = Chem.RemoveHs(conn_mol)
        smi = Chem.MolToSmiles(cm)
        mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        mol = None
        smi = None
        print(f"Error in generating mol from xyz: {e}")

    return mol, smi


def str_has_numbers(string: str) -> bool:
    return any(char.isdigit() for char in string)


def split_str_number(string: str) -> str:
    head = string.rstrip("0123456789")
    tail = int(string[len(head) :])
    return head, tail


def write_xyz_files_to_dict(
    xyz_folder=(
        "/groups/kemi/borup/pKalculator/data/external/"
        "shen_et_al/supp_info_shen_et_al.txt"
    ),
):
    """

    Returns:
        dict: dictionary for evey compound with xyz coordinates
    """
    record_lines = False
    lst_lines = []
    with open(xyz_folder, "r") as f:
        for i, line in enumerate(f.readlines()):
            if not record_lines:
                if line.startswith("Table S2"):
                    # if line.startswith("VIBRATIONAL FREQUENCIES"):
                    record_lines = True
            elif line.startswith("Table S3"):
                record_lines = False
            else:
                lst_lines.append(line)

    table_start_pattern = re.compile(r"[A-Z][0-9]+[-][0-9]+|[A-Z][0-9]+")
    lst_match = [
        (idx, re.match(table_start_pattern, line))
        for idx, line in enumerate(lst_lines)
        if re.match(table_start_pattern, line) is not None
    ]
    lst_idx = [idx for (idx, match) in lst_match]
    periodic_table = Chem.GetPeriodicTable()

    dict_compounds = {}
    # Iterate over lst_idx and use it as index to lst_lines
    for i, idx in enumerate(lst_idx):
        try:
            for ix, match in lst_match:
                if ix == idx:
                    matche = match.group(0)
            lst_xyz = []
            if idx != lst_idx[-1]:
                for line in lst_lines[idx + 1 : lst_idx[i + 1] - 2]:
                    if str_has_numbers(line):
                        symbol = periodic_table.GetElementSymbol(int(line.split()[1]))
                        lst_atom_xyz = [symbol] + [
                            item
                            for (idx_line, item) in enumerate(line.split())
                            if idx_line > 2
                        ]
                        lst_xyz.append(lst_atom_xyz)
            dict_compounds[matche] = lst_xyz

            if idx == lst_idx[-1]:
                lst_xyz = []
                for line in lst_lines[idx + 1 :]:
                    if str_has_numbers(line):
                        symbol = periodic_table.GetElementSymbol(int(line.split()[1]))
                        lst_atom_xyz = [symbol] + [
                            item
                            for (idx_line, item) in enumerate(line.split())
                            if idx_line > 2
                        ]
                        lst_xyz.append(lst_atom_xyz)
                dict_compounds[matche] = lst_xyz
        except Exception as e:
            logging.error(f"error in {idx}", exc_info=e)
            print(f"error in {idx}")
            continue

    return dict_compounds


def write_xyz_files(folder_path):
    """write xyz files to folder from a dictionary of xyz coordinates

    Args:
        folder_path (str): _description_
    """
    # /groups/kemi/borup/pKalculator/data/external/shen_et_al/xyz_files
    for key, value in write_xyz_files_to_dict().items():
        with open(f"{folder_path}/{key}.xyz", "w") as file:
            file.write(str(len(value)) + "\n\n")
            file.writelines("\t".join(i) + "\n" for i in value)


def read_xyz_to_df(xyz_folder: str) -> pd.DataFrame:
    """_summary_

    Args:
        xyz_folder (str): folder location,
        e.g '/groups/kemi/borup/pKalculator/data/external/shen_et_al/xyz_files'

    Returns:
        pd.DataFrame: _description_
    """

    dict_mols_smiles = {
        split_str_number(file.name.split(".")[0]): get_xyz_to_mol(file, charge=0)
        for pattern in ("[A-Z][0-9].xyz", "[A-Z][0-9][0-9].xyz")
        for file in Path(xyz_folder).glob(pattern)
    }

    # correct wrong smiles from xyz files: C11
    # add missing xyz files: A13, A14, A15
    dict_mols_smiles[("A", 13)] = (
        Chem.MolFromSmiles("CC(C)(C)OC(=O)n1cccc1"),
        "CC(C)(C)OC(=O)n1cccc1",
    )
    dict_mols_smiles[("A", 14)] = (
        Chem.MolFromSmiles("CC(C)(C)OC(=O)n1cccn1"),
        "CC(C)(C)OC(=O)n1cccn1",
    )
    dict_mols_smiles[("A", 15)] = (
        Chem.MolFromSmiles("CC(C)(C)OC(=O)n1ccnc1"),
        "CC(C)(C)OC(=O)n1ccnc1",
    )
    dict_mols_smiles[("C", 11)] = (
        Chem.MolFromSmiles("Cn1cnc2ccccc21"),
        "Cn1cnc2ccccc21",
    )

    # sort dict by key
    sorted_dict_mols_smiles = dict(sorted(dict_mols_smiles.items(), key=lambda x: x[0]))

    df_smiles = pd.DataFrame(
        [
            (key[0] + str(key[1]), value[1])
            for key, value in sorted_dict_mols_smiles.items()
        ],
        columns=["name", "smiles"],
    )

    # insert comment column
    df_smiles.loc[[12, 13, 14], "comment"] = "no xyz file"
    df_smiles.loc[48, "comment"] = "wrong smiles from xyz file. corrected"

    return df_smiles


def calc_pKa_paper(
    df_neutral="",
    df_anion="",
    ref_neutral="a1",
    ref_anion="a1-1",
    neutral=None,
    anion=None,
    pKa_ref=35,
    T=None,
):
    """
    Het-H + Furan- --> Het- + Furan
    """
    if T is None:
        T = 298.15  # K

    gas_constant = 1.987  # kcal/(mol*K) or 8.31441 J/(mol*K) gas constant
    conversion_factor = Hartree * mol / kcal  # convert energy from Hartree to kcal/mol

    idx_neutral_ref = df_neutral.index[df_neutral["Neutral"] == ref_neutral][0]
    idx_anion_ref = df_anion.index[df_anion["Anions"] == ref_anion][0]

    G_neutral_ref = (
        df_neutral.at[idx_neutral_ref, "G(gas)=TCG+E"] * conversion_factor
        + df_neutral.at[idx_neutral_ref, "dG(sol)"]
    )
    G_anion_ref = (
        df_anion.at[idx_anion_ref, "G(gas)=TCG+E"] * conversion_factor
        + df_anion.at[idx_anion_ref, "dG(sol)"]
    )

    idx_neutral = df_neutral.index[df_neutral["Neutral"] == neutral][0]
    idx_anion = df_anion.index[df_anion["Anions"] == anion][0]

    G_neutral = (
        df_neutral.at[idx_neutral, "G(gas)=TCG+E"] * conversion_factor
        + df_neutral.at[idx_neutral, "dG(sol)"]
    )
    G_anion = (
        df_anion.at[idx_anion, "G(gas)=TCG+E"] * conversion_factor
        + df_anion.at[idx_anion, "dG(sol)"]
    )

    dG_exchange = G_neutral_ref + G_anion - G_anion_ref - G_neutral

    pKa_paper = round(
        pKa_ref + (dG_exchange / (gas_constant * T * np.log(10) / 1000)), 1
    )

    return pKa_paper


def get_pKa_from_entry(df_neutral="", df_anion="", neutral=""):
    """
    Returns a tuple of anions and the pKa values calculated from the common neutral compound
    """
    idx_anions = df_anion.index[
        df_anion["Anions"].str.contains(neutral + "-", regex=False)
    ].tolist()

    pKas = [
        calc_pKa_paper(
            df_neutral=df_paper_neutral,
            df_anion=df_paper_anions,
            ref_neutral="a1",
            ref_anion="a1-2",
            neutral=neutral,
            anion=df_anion.Anions[idx],
            pKa_ref=35,
            T=None,
        )
        for idx in idx_anions
    ]
    anion_names = [df_anion.Anions[idx] for idx in idx_anions]
    tup_names_pKas = tuple(zip(anion_names, pKas))

    return tup_names_pKas


if __name__ == "__main__":
    import time

    start = time.perf_counter()

    path_data = Path(Path.home() / "pKalculator" / "data" / "external" / "shen_et_al")
    df_paper_neutral = pd.read_csv(
        path_data / "data_paper_neutral.csv"
    )  # pd.read_excel(path_data / "data_paper_neutral.ods", engine="odf")
    df_paper_anions = pd.read_csv(
        path_data / "data_paper_anions.csv"
    )  # pd.read_excel(path_data / "data_paper_anions.ods", engine="odf")

    paper_data = list(
        itertools.chain(
            *[
                get_pKa_from_entry(
                    df_neutral=df_paper_neutral, df_anion=df_paper_anions, neutral=name
                )
                for name in df_paper_neutral.Neutral
            ]
        )
    )
    df_paper_data = pd.DataFrame(paper_data, columns=["name_paper", "pka_paper"])
    df_paper_data.to_csv(path_data / "pka_shen_et_al_paper_test.csv", index=False)

    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} second(s)")
