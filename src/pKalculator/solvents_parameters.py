from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
import itertools


def smi_to_descriptors(smile):
    # descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    descriptor_names = ['exactmw', 'lipinskiHBA', 'lipinskiHBD', 'NumHBA', 'NumHBD', 'CrippenClogP']
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    mol = Chem.MolFromSmiles(smile)
    descriptors = []
    if mol:
        descriptors = np.array(get_descriptors.ComputeProperties(mol))
    return descriptors


# Create a dataframe with the descriptors and
def create_descriptors_df(df):
    # descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    descriptor_names = ['exactmw', 'lipinskiHBA', 'lipinskiHBD', 'NumHBA', 'NumHBD', 'CrippenClogP']
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)

    descriptors_df = pd.DataFrame(columns=descriptor_names)
    for idx, row in df.iterrows():
        descriptors = smi_to_descriptors(row['smiles'])
        descriptors_df.loc[idx] = descriptors
    return descriptors_df


def gen_solvent_df():
    # Generate smiles for solvent_name
    #  solvent_smiles = {
    #             'Water' : 'O', @
    #             'DMSO' : 'CS(=O)C', @
    #             'DMF' : 
    #             'Acetonitrile' : 'CC#N', @
    #             'Methanol' : 'CO',
    #             'Ethanol' : 'CCO',
    #             'Acetone' : 'CC(=O)C',
    #             'Pyridine' : 'c1ccncc1',
    #             'dichloromethane' : , 
    #             'THF' : 'C1CCOC1', @
    #             'Ether' : 'CCOCC', @
    #             'Toluene' : 'Cc1ccccc1', @
    #             '1,4-Dioxane' : 'C1COCCO1',
    #             'Heptane' : 'CCCCCCC'
    #             }
    solvent_smiles = {
                'Water' : 'O',
                'DMSO' : 'CS(=O)C', 
                'Acetonitrile' : 'CC#N', 
                'Acetone' : 'CC(=O)C', 
                'CH2Cl2' : 'C(Cl)Cl',  
                'THF' : 'C1CCOC1', 
                'Ether' : 'CCOCC', 
                'Toluene' : 'Cc1ccccc1', 
                }

    # Create a dataframe from solvent_smiles
    df_solvent = pd.DataFrame(list(solvent_smiles.items()), columns=['solvent', 'smiles'])

    # Create a dataframe with the descriptors and insert into the solvent_smiles_df
    df_descriptors = create_descriptors_df(df_solvent)
    df_solvent = pd.concat([df_solvent, df_descriptors], axis=1)

    return df_solvent


def find_all_combinations(iter_dict, df_solvent):

    iter_dict['solvent_name'] = tuple(df_solvent['solvent'])
    # find all combinations in iter_xtb_dict
    all_combinations = []
    for key, value in iter_dict.items():
        all_combinations.append(value)
    all_combinations = list(itertools.product(*all_combinations))
    df_iter_xtb = pd.DataFrame(all_combinations, columns=iter_dict.keys())
    return df_iter_xtb


dict_iter_xtb = {
            'method' : (1, 2), 
            'solvent_model' : ('gbsa', 'alpb'),
            'solvent_name' : None
            }

if __name__ == "__main__":

    df_solvent = gen_solvent_df()
    df_iter_xtb = find_all_combinations(iter_dict=dict_iter_xtb, df_solvent=df_solvent)

    for idx, row in df_iter_xtb.iterrows():
            xtb_params = {
                'method' : row['method'], 
                'solvent_model' : row['solvent_model'],
                'solvent_name' : row['solvent_name']
                }
            print(xtb_params)