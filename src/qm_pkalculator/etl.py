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

sys.path.insert(0, "/lustre/hpc/kemi/borup/pKalculator/src/smi2gcs'")
# FIX THIS WHEN WORKING IN ANOTHER DIRECTORY
from DescriptorCreator.PrepAndCalcDescriptor import Generator


def load_pickles(folder):
    # print(folder)
    return pd.concat(
        [
            pd.read_pickle(pkl)[1].assign(file=pkl.name)
            for pkl in folder.glob("*_result.pkl")
            if type(pd.read_pickle(pkl)[1]) != str
        ],
        ignore_index=True,
    )  # .sort_values(by='file')


def compute_relative_energy(row, energy_col):
    return [
        (
            float("inf")
            if val == float("inf") or row[energy_col] == float("inf")
            else (val - row[energy_col])
        )
        for val in row[f"lst_{energy_col}_deprot"]
    ]


def process_submitted_files(path_submitit: str, prelim_path=None) -> pd.DataFrame:
    if not Path(path_submitit).is_dir():
        raise ValueError("path is not a directory")

    if prelim_path:
        try:
            df_prelim = pd.read_pickle(prelim_path)
        except FileNotFoundError:
            raise ValueError(f"{prelim_path} is not a valid path to a pickle file")

    # df_prelim = pd.read_pickle('/groups/kemi/borup/pKalculator/data/prelim/df_prelim_20231130.pkl')
    df_submitted = load_pickles(path_submitit)

    if prelim_path:
        missing_names = set(df_prelim.names.unique()).difference(
            set(df_submitted.names.unique())
        )
        df_submitted = pd.concat(
            [df_submitted, df_prelim.loc[df_prelim.names.isin(missing_names)]]
        )

    df_submitted["failed_xtb"] = df_submitted["e_xtb"].isnull()
    df_submitted["failed_dft"] = df_submitted["e_dft"].isnull()
    df_submitted["e_xtb"] = df_submitted["e_xtb"].fillna(float("inf"))
    df_submitted["e_dft"] = df_submitted["e_dft"].fillna(float("inf"))
    # df_submitted.rename(columns={"lst_atomsite": "atomsite"}, inplace=True)
    # df_submitted.rename(columns={"lst_atomindex": "atom_index"}, inplace=True)
    df_submitted.rename(columns={"atom_index": "atomindex"}, inplace=True)
    df_submitted["atomindex"] = df_submitted["atomindex"].astype(int)
    # df_submitted["smiles_deprot"] = df_submitted["smiles_deprot_map"]

    df_neutral = df_submitted.query("neutral == True").reset_index(drop=True)

    df_neg = (
        df_submitted.query("neutral == False")
        .groupby("ref_name")
        .agg(
            {
                col: lambda x: x.tolist()
                for col in df_submitted.columns
                if col
                not in [
                    "ref_name",
                    "pka_exp",
                    "comment",
                    "outlier_note",
                ]  # check here
            }
        )
        .rename(
            columns=lambda x: (
                f"lst_{x}_deprot"
                if x != "ref_mol_smiles_map"
                else "lst_ref_mol_smiles_map"
            )
        )
    )

    # lst_smiles_deprot_map_deprot, 	lst_smiles_deprot_deprot

    # Sort the lists within each row based on 'lst_atomsite'
    for i, row in df_neg.iterrows():
        sorted_idx = [
            idx
            for idx, _ in sorted(enumerate(row.lst_atomsite_deprot), key=lambda x: x[1])
        ]
        for col in df_neg.columns:
            if col.startswith("lst_"):
                df_neg.at[i, col] = [row[col][idx] for idx in sorted_idx]

    df_neutral_merged = pd.merge(
        df_neutral, df_neg, on="ref_name", how="left", suffixes=("", "_new")
    )

    df_neutral_merged["lst_e_rel_xtb"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_xtb",)
    )
    df_neutral_merged["lst_e_rel_dft"] = df_neutral_merged.apply(
        compute_relative_energy, axis=1, args=("e_dft",)
    )

    df_neutral_merged["e_rel_min_xtb"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_xtb"]), axis=1
    )

    df_neutral_merged["e_rel_min_dft"] = df_neutral_merged.apply(
        lambda row: min(row["lst_e_rel_dft"]), axis=1
    )

    df_neutral_merged["lst_pka_lfer"] = df_neutral_merged.lst_e_rel_dft.apply(
        lambda x: [round(calc_pka_lfer(e_rel), 4) for e_rel in x]
    )
    df_neutral_merged["pka_min_lfer"] = df_neutral_merged.lst_pka_lfer.apply(
        lambda x: min(x)
    )

    df_neutral_merged[f"atomsite_min_lfer"] = df_neutral_merged.apply(
        lambda row: (
            -1
            if row["pka_min_lfer"] == float("inf")
            else [
                atomsite
                for atomsite, pka in zip(
                    row["lst_atomsite_deprot"], row["lst_pka_lfer"]
                )
                if pka == row["pka_min_lfer"]
            ][0]
        ),
        axis=1,
    )

    drop_cols = [
        "smiles_deprot_map",
        "atomsite",
        "atomindex",
        "ref_mol_smiles_map",
        "lst_gfn_method_deprot",
        "lst_solvent_model_deprot",
        "lst_solvent_name_deprot",
        "smiles_neutral",
    ]
    rename_cols = {
        "mol": "mol_neutral",
        "e_xtb": "e_xtb_neutral",
        "e_dft": "e_dft_neutral",
        "failed_xtb": "failed_xtb_neutral",
        "failed_dft": "failed_dft_neutral",
        "lst_smiles_deprot_map_deprot": "lst_smiles_map_deprot",
    }
    df_neutral_merged.drop(columns=drop_cols, inplace=True)
    df_neutral_merged.rename(columns=rename_cols, inplace=True)

    return df_neutral_merged


def get_cm5_desc_vector(smi_name, smi, lst_atom_index, n_shells=6):
    # make sure that smi is canonical. Smiles in our workflow should provide a canonical smiles as Chen.MolToSmiles() by default generates the canonical smiles
    generator = Generator()
    des = (
        "GraphChargeShell",
        {"charge_type": "cm5", "n_shells": n_shells, "use_cip_sort": True},
    )
    try:
        cm5_list = generator.calc_CM5_charges(
            smi=smi, name=smi_name, optimize=False, save_output=True
        )
        (
            atom_indices,
            descriptor_vector,
            mapper_vector,
        ) = generator.create_descriptor_vector(lst_atom_index, des[0], **des[1])
    except Exception:
        descriptor_vector = None

    return cm5_list, atom_indices, descriptor_vector, mapper_vector


def calc_pka_lfer(e_rel: float) -> float:
    # pka = 0.5941281 * e_rel - 159.33107321
    pka = 0.59454292 * e_rel - 159.5148093
    return pka


def pka_dmso_to_pka_thf(pka: float, reverse=False) -> float:
    pka_thf = -0.963 + 1.046 * pka
    if reverse:
        pka_dmso = (pka_thf + 0.963) / 1.046
        return pka_dmso

    return pka_thf


if __name__ == "__main__":
    path_submitit = Path.home() / "pKalculator/data/raw/misc/submitit/"
    folder_submitit = path_submitit / "submitit_test"
    path_prelim = Path.home() / "pKalculator/data/prelim"
    path_calc = Path.home() / "pKalculator/data/raw/misc/calc/calc_test"

    df_results = process_submitted_files(
        path_submitit=Path(f"{path_submitit / folder_submitit}"),
        prelim_path=f"{path_prelim / 'df_prelim_calc_test.pkl'}",
    )

    # for data
    # df_results = pd.read_pickle(
    #     "/groups/kemi/borup/pKalculator/data/processed/misc/full_dataset_prelim_20231221.pkl"
    # )
    # reaxys data
    # df_results = pd.read_pickle(
    #     "/groups/kemi/borup/pKalculator/data/processed/reaxys/ML_reaxys_all_reactions_20240206.pkl"
    # )

    # icb data
    # df_results = pd.read_pickle(
    #     "/groups/kemi/borup/pKalculator/data/prelim/ML_prelim_icb_thf_all_20240223.pkl"
    # )
    print("calculating cm5 charges and descriptor vectors")
    output = df_results.apply(
        lambda row: get_cm5_desc_vector(
            smi_name=row["names"],
            smi=row["smiles"],
            lst_atom_index=row[
                "lst_atomindex_deprot"
            ],  # lst_atom_index , lst_atom_index_deprot,
            n_shells=6,
        ),
        axis=1,
    )
    (
        df_results["cm5"],
        df_results["atom_indices"],
        df_results["descriptor_vector"],
        df_results["mapper_vector"],
    ) = zip(*output)

    print("saving results to pickle")
    # save df_results
    df_results.to_pickle(
        Path(
            Path.home()
            / "pKalculator"
            / "data"
            / "processed"
            / "misc"
            / "df_results_test2.pkl"
        )
    )
