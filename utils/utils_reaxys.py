from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import pandas as pd

aldol_rxn = "[#6:1](=[#8:2])[#6:3].[#6:4](=[#8:5])>>[#6:1](=[#8:2])[#6:3][#6:4]-[#8:5]"
michael_rxn = "[#6,#8:1](=[#8:2])[#6:3].[#6:4]=[#6:5][#6:6](=[#8:7])>>[#6,#8:1](=[#8:2])[#6:3][#6:4][#6:5][#6:6](=[#8:7])"
michael_rxn2 = "[#6,#8:1](=[#8:2])[#6:3].[#6:4]=[#6:5][NX3+:6](=[#8:7])[O-:8]>>[#6,#8:1](=[#8:2])[#6:3][#6:4][#6:5][NX3+:6](=[#8:7])[O-:8]"
claisen_rxn = (
    "[#6:1](=[#8:2])[#6:3].[#6:4](=[#8:5])[#8]>>[#6:1](=[#8:2])[#6:3][#6:4](=[#8:5])"
)
claisen_rxn2 = (
    "[#6:1](=[#8:2])[#6:3].[#6:4](=[#8:5])[#8]>>[#6R:1](=[#8:2])[#6R:3][#6R:4](=[#8:5])"
)


def find_group(smiles, group_smarts):
    try:
        smarts = Chem.MolFromSmarts(group_smarts)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        return rdkit_mol.HasSubstructMatch(smarts)
    except Exception as e:
        print(f"Error: {e}")
        return False


def remove_groups(df):
    # iterate over df and create mol objects for each compound.
    # Check if compound is an amine or alcohol and add True/False to df
    amines = "[NX3;H2,H1;!$(NC=O)]"
    alcohol = "[#6][OX2H]"
    # For standard amide smarts: [NX3;H2,H1][CX3](=[OX1])[#6]
    amides = "[NX3;H2,H1][CX3](=[OX1])[#6]"
    df["amine"] = df["enolate_smi"].apply(find_group, group_smarts=amines)
    df["alcohol"] = df["enolate_smi"].apply(find_group, group_smarts=alcohol)
    df["amide"] = df["enolate_smi"].apply(find_group, group_smarts=amides)

    # remove amines and alcohols
    # df_pKa_BordwellCH = df_pKa_BordwellCH[(df_pKa_BordwellCH['amine'] == False) &
    # (df_pKa_BordwellCH['alcohol'] == False)].reset_index(drop=True)
    df_new = df.query(
        "amine == False and alcohol == False and amide == False"
    ).reset_index(drop=True)

    return df_new


def get_unique_reps(rdkit_smis):
    lst_unique_smiles = []
    for rdkit_smi in rdkit_smis:
        try:
            rdkit_mol = Chem.MolFromSmiles(rdkit_smi)
            rdkit_smi = Chem.MolToSmiles(rdkit_mol, isomericSmiles=True)
            lst_unique_smiles.append(rdkit_smi)
        except Exception as e:
            print(f"Could not convert SMILES: {rdkit_smi}")
            continue

    return lst_unique_smiles


def get_df_from_reaxys(df_csv):
    lst_rxn = []
    lst_reagents = []

    # Groupby 'Reaction ID' and apply the following operations to each group
    grouped_reactions = df_csv.dropna(subset=["Reaction"]).groupby("Reaction ID")
    for k, reaction_group in grouped_reactions:
        # Extract the 'Reaction' column from the group
        reaction_smiles = reaction_group["Reaction"].values[0]
        if type(reaction_group["Reagent"].values[0]) == str:
            try:
                lst_reagent = reaction_group["Reagent"].values[0].split(";")
                lst_reagents.append((k, [reagent.strip() for reagent in lst_reagent]))
            except Exception as e:
                continue
        if not reaction_smiles:
            continue

        # Split reaction SMILES (reaction_smiles) into reactants and products
        rxn_reactants, rxn_products = reaction_smiles.split(">>")
        if len(rxn_reactants) == 0 or len(rxn_products) == 0:
            continue
        if "*" in rxn_reactants or "*" in rxn_products:
            continue

        lst_rxn_reactants = set(get_unique_reps(rxn_reactants.split(".")))
        lst_rxn_products = set(get_unique_reps(rxn_products.split(".")))
        # print(get_unique_reps(rxn_products.split('.')))

        rxn = ".".join(lst_rxn_reactants) + ">>" + ".".join(lst_rxn_products)
        if "*" in rxn:
            continue

        lst_rxn.append((k, rxn))

    # Convert the list of tuples to a DataFrame directly
    df_rxns = pd.DataFrame(lst_rxn, columns=["rxn_id", "rxn_smiles"])

    df_rxns["reactants"] = (
        df_rxns["rxn_smiles"].str.split(">>").str[0].str.split(".").apply(tuple)
    )
    df_rxns["products"] = (
        df_rxns["rxn_smiles"].str.split(">>").str[1].str.split(".").apply(tuple)
    )
    # print(df_rxns['rxn_smiles'].str.split('>>').str[1].str.split('.').values)
    # Sort and drop duplicates based on 'rxn_id' column
    df_rxns = (
        df_rxns.drop_duplicates(subset=["rxn_id"])
        .sort_values(by="rxn_id")
        .reset_index(drop=True)
    )
    # merge df_rxns and df_reagents by rxn_id
    df_reagents = pd.DataFrame(lst_reagents, columns=["rxn_id", "reagents"])
    df_rxns = df_rxns.merge(df_reagents, on="rxn_id", how="left")
    return df_rxns


def get_top_n_reagents(df, top_n=5):
    lst_all_reagents = [
        reagent.strip()
        for reagents in df.reagents
        if type(reagents) != float
        for reagent in reagents
    ]
    occurrence = {item: lst_all_reagents.count(item) for item in lst_all_reagents}
    occurrence_sorted = {
        k: v
        for k, v in sorted(occurrence.items(), key=lambda item: item[1], reverse=True)
    }
    # Sort the dictionary items by their values (i.e. the number of occurrences)
    sorted_items = sorted(occurrence_sorted.items(), key=lambda x: x[1], reverse=True)

    # Get the 5 most occurring items
    most_occuring = sorted_items[:top_n]

    # Extract the names of the most occurring items
    # most_occuring_names = [name for name, count in most_occuring]

    df_most_occuring = pd.DataFrame(
        data=most_occuring, columns=["reagents", "occurrence"]
    )

    # Calculate the total number of occurrences of the most occurring reagents
    total_occurrences = df_most_occuring["occurrence"].sum()

    # Calculate the percentage of occurrence of each reagent
    df_most_occuring["percentage"] = df_most_occuring["occurrence"] / total_occurrences

    df_most_occuring = df_most_occuring.sort_values(
        by="percentage", ascending=True
    ).reset_index(drop=True)

    return occurrence_sorted, df_most_occuring


def create_unique_df_enolates(df_filtered_enolates, df_most_occuring):
    lst_unique_dfs = []
    for reagent in df_most_occuring.reagents:
        temp_df = df_filtered_enolates.loc[
            df_filtered_enolates["reagents"].apply(
                lambda x: any(item in x for item in [reagent] if type(x) != float)
            )
        ].drop_duplicates(subset=["enolate_smi"], keep="first")
        temp_df = remove_groups(temp_df)
        lst_unique_dfs.append(temp_df)

    df_unique_enolates = pd.concat(lst_unique_dfs)
    df_unique_enolates.drop_duplicates(
        subset=["enolate_smi"], keep="first", inplace=True
    )

    return df_unique_enolates


def remove_stereoinfo(rdkit_mol):
    for atom in rdkit_mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    return rdkit_mol


def get_mapped_reaction_site(rxn, product, reactants):
    if type(rxn) == str:
        rxn = rdChemReactions.ReactionFromSmarts(rxn)

    for i, m in enumerate(reactants):
        for atom in m.GetAtoms():
            atom.SetIntProp("reactant_idx", i)

    lst_changed_atoms = []
    for atom in product.GetAtoms():
        if atom.GetPropsAsDict().get("_ReactionDegreeChanged"):
            lst_changed_atoms.append(
                (
                    atom.GetPropsAsDict().get("old_mapno"),
                    atom.GetPropsAsDict().get("react_atom_idx"),
                    atom.GetIdx(),
                )
            )

    # print(f'lst_changed_atoms: {lst_changed_atoms}')

    atomMapToReactantMap = {}
    for ri in range(rxn.GetNumReactantTemplates()):
        rt = rxn.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            for di in lst_changed_atoms:
                if atom.GetAtomMapNum() == di[0]:
                    atomMapToReactantMap[ri] = di

    for idx, reactant in enumerate(reactants):
        for atom in reactant.GetAtoms():
            if atom.GetIdx() == atomMapToReactantMap[idx][1]:
                atom.SetAtomMapNum(atomMapToReactantMap[idx][0])

    reactant_smiles_mapped = [Chem.MolToSmiles(x) for x in reactants][0]

    for atom in product.GetAtoms():
        if atom.GetIdx() == atomMapToReactantMap[idx][2]:
            atom.SetAtomMapNum(atomMapToReactantMap[idx][0])

    product_smiles_mapped = Chem.MolToSmiles(product)
    product_mol_mapped = Chem.MolFromSmiles(product_smiles_mapped)
    if product_mol_mapped is None:
        return None, None

    # display(Chem.MolFromSmiles(reactant_smiles_mapped[0]))
    # display(product_mol_mapped)
    return reactant_smiles_mapped, product_smiles_mapped


def run_rxn_no_stereo(rxn_smiles_type, smi_reactant1, smi_reactant2, smi_product):
    smi_enolate = ""
    mapped_smiles = None, None
    reactant1 = remove_stereoinfo(Chem.MolFromSmiles(smi_reactant1))
    reactant2 = remove_stereoinfo(Chem.MolFromSmiles(smi_reactant2))
    product_mol = remove_stereoinfo(Chem.MolFromSmiles(smi_product))
    smi_product = Chem.MolToSmiles(product_mol)

    rxn = AllChem.ReactionFromSmarts(rxn_smiles_type)
    ps = rxn.RunReactants((reactant1, reactant2))
    ps2 = rxn.RunReactants((reactant2, reactant1))
    if any(x for x in [Chem.MolToSmiles(p[0]) for p in ps] if x == smi_product):
        smi_enolate = smi_reactant1
        # print([x for x in [Chem.MolToSmiles(p[0]) for p in ps] if x == smi_product])
        if (
            len([x for x in [Chem.MolToSmiles(p[0]) for p in ps] if x == smi_product])
            > 1
        ):
            print("Warning: multiple matches")
        # print(f'reactant1 is enolate: {smi_enolate}')
        # print('getting mapped enolate_smiles')
        lst_correct_products = [
            p[0] for p in ps if Chem.MolToSmiles(p[0]) == smi_product
        ]
        for product in lst_correct_products:
            temp_mapped_smiles = get_mapped_reaction_site(
                rxn=rxn, product=product, reactants=[x for x in (reactant1,)]
            )
            if None in temp_mapped_smiles:
                # print('No mapping found')
                continue
            else:
                mapped_smiles = temp_mapped_smiles
    elif any(x for x in [Chem.MolToSmiles(p[0]) for p in ps2] if x == smi_product):
        smi_enolate = smi_reactant2
        # print([x for x in [Chem.MolToSmiles(p[0]) for p in ps2] if x == smi_product])
        if (
            len([x for x in [Chem.MolToSmiles(p[0]) for p in ps2] if x == smi_product])
            > 1
        ):
            print("Warning: multiple matches")
        # print(f'reactant2 is enolate: {smi_enolate}')
        # print('getting mapped enolate_smiles')
        lst_correct_products = [
            p[0] for p in ps2 if Chem.MolToSmiles(p[0]) == smi_product
        ]
        for product in lst_correct_products:
            temp_mapped_smiles = get_mapped_reaction_site(
                rxn=rxn, product=product, reactants=[x for x in (reactant2,)]
            )
            if None in temp_mapped_smiles:
                # print('No mapping found')
                continue
            else:
                mapped_smiles = temp_mapped_smiles
    # print(mapped_smiles)
    reactant_smiles_mapped, product_smiles_mapped = mapped_smiles
    # print(f'mapped reactant smiles: {reactant_smiles_mapped},
    # mapped product smiles: {product_smiles_mapped}')
    return smi_enolate, reactant_smiles_mapped, product_smiles_mapped


def get_enolate_smi(rxn_smiles_type="", rxn_smiles=""):
    if rxn_smiles == "":
        print("Error: rxn_smiles is empty")
        return ""
    smi_reactants = rxn_smiles.split(">>")[0].split(".")

    if len(smi_reactants) < 2:
        smi_reactant1 = smi_reactants[0]
        smi_reactant2 = smi_reactants[0]
    else:
        smi_reactant1 = smi_reactants[0]
        smi_reactant2 = smi_reactants[1]

    smi_product = rxn_smiles.split(">>")[1]

    if smi_product.count(".") >= 1:
        lst_smi_product = smi_product.split(".")
        (
            enolate_smi,
            reactant_smiles_mapped,
            product_smiles_mapped,
        ) = run_rxn_no_stereo(
            rxn_smiles_type=rxn_smiles_type,
            smi_reactant1=smi_reactant1,
            smi_reactant2=smi_reactant2,
            smi_product=lst_smi_product[0],
        )
        if enolate_smi == "":
            (
                enolate_smi,
                reactant_smiles_mapped,
                product_smiles_mapped,
            ) = run_rxn_no_stereo(
                rxn_smiles_type=rxn_smiles_type,
                smi_reactant1=smi_reactant1,
                smi_reactant2=smi_reactant2,
                smi_product=lst_smi_product[1],
            )
    else:
        (
            enolate_smi,
            reactant_smiles_mapped,
            product_smiles_mapped,
        ) = run_rxn_no_stereo(
            rxn_smiles_type=rxn_smiles_type,
            smi_reactant1=smi_reactant1,
            smi_reactant2=smi_reactant2,
            smi_product=smi_product,
        )

    return enolate_smi, reactant_smiles_mapped, product_smiles_mapped


def reorder_smiles(smiles, smiles_template):
    old_atommap = []
    new_atommap = []

    mol = Chem.MolFromSmiles(smiles)
    mol_template = Chem.MolFromSmiles(smiles_template)
    assert mol.GetNumAtoms() == mol_template.GetNumAtoms()
    atom_matches = mol.GetSubstructMatch(mol_template)
    assert len(atom_matches) == mol_template.GetNumAtoms()
    mol_reordered = Chem.RenumberAtoms(mol, atom_matches)
    for idx, atom in enumerate(mol_reordered.GetAtoms()):
        if atom.HasProp("molAtomMapNumber"):
            old_atommap.append(atom.GetProp("molAtomMapNumber"))
            new_atommap.append(idx + 1)
        atom.SetAtomMapNum(idx + 1)
    smiles_reordered = Chem.MolToSmiles(mol_reordered)
    return smiles_reordered, old_atommap, new_atommap


if __name__ == "__main__":
    """Example usage of the functions in this module"""
    df_bordwell = pd.read_csv(
        "/groups/kemi/borup/pKalculator/data/external/bordwell/bordwellCH_rmb_v3.exp"
    )
    df_aldol = pd.read_csv(
        "/groups/kemi/borup/pKalculator/data/external/reaxys/reaxys_aldol_reaction.tsv",
        delimiter="\t",
    )
    print(f"Shape of DataFrame: {df_aldol.shape}")
    print(f'Unique reactions in DataFrame: {len(set(list(df_aldol["Reaction ID"])))}')
    df_aldol_rxns = get_df_from_reaxys(df_csv=df_aldol)
    occurrence_sorted_aldol, df_most_occuring_aldol = get_top_n_reagents(
        df=df_aldol_rxns, top_n=5
    )
    df_most_occuring_aldol = df_most_occuring_aldol.query(
        'reagents != "titanium tetrachloride"'
    ).reset_index(drop=True)

    df_aldol_rxns["enolate_smi"] = df_aldol_rxns["rxn_smiles"].apply(
        lambda x: get_enolate_smi(rxn_smiles_type=aldol_rxn, rxn_smiles=x)
    )
    df_aldol_unique_enolates = create_unique_df_enolates(
        df_filtered_enolates=df_aldol_rxns.query("enolate_smi != ''"),
        df_most_occuring=df_most_occuring_aldol,
    )
    # remove smiles that are in the bordwell dataset
    df_aldol_unique_enolates = df_aldol_unique_enolates.loc[
        df_aldol_unique_enolates.enolate_smi.apply(
            lambda x: x in df_bordwell.smiles.values
        )
        == False
    ]
    print(df_aldol_unique_enolates)
