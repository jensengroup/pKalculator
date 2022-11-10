import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_pickles(folder):
   return pd.concat([pd.read_pickle(pkl)[1].assign(file=pkl.name) 
            for pkl in folder.glob('*_result.pkl')
            ], ignore_index=True)#.sort_values(by='file')


def calc_pKa(df, row_idx, pKa_ref=35, T=None):
    """
    Het-H + Furan- --> Het- + Furan
    """
    import ast
    if T == None:
        T = 298.15 #K
    
    gas_constant = 1.987 #kcal/(mol*K) or 8.31441 J/(mol*K) gas constant
    # conversion_factor = Hartree * mol/kcal #convert energy from Hartree to kcal/mol

    idx_ref = df.index[df['name'] == 'furan'][0]
    # idx_deprot = ast.literal_eval(df.at[idx_ref, 'lst_name_deprot']).index('furan#-1=2')
    idx_deprot = df.at[idx_ref, 'lst_name_deprot'].index('furan#-1=2')
    # print(idx_deprot)
    #take the negative as the relative energy is calculated as Furan --> Furan-
    E_rel_ref = - df.at[idx_ref, 'lst_E_rel'][idx_deprot]

    # row_idx = 0
    lst_E_rel = df.at[row_idx, 'lst_E_rel']
 
    lst_pKa = []
    for E_rel in lst_E_rel:
        dG_exchange = E_rel_ref + E_rel
        pKa = round(pKa_ref + (dG_exchange/(gas_constant*T*np.log(10)/1000)), 1)
        lst_pKa.append(pKa)

    return lst_pKa
    

if __name__ == "__main__":
    path_data = Path(Path.home()/'pKalculator'/'data')
    
    folders = ['submitit_pKalculator_1_alpb_DMSO_1', 'submitit_pKalculator_2_alpb_DMSO_1', 'submitit_pKalculator_1_gbsa_DMSO_1', 'submitit_pKalculator_2_gbsa_DMSO_1']

    lst_df_result = []
    for folder in folders:
        path = Path(Path.home()/'pKalculator'/'src'/'pKalculator'/folder)
        df_results = load_pickles(folder=path)
        for idx, row in df_results.iterrows():
            lst_pKa = calc_pKa(df=df_results, row_idx=idx, pKa_ref=35, T=None)
            df_results.at[idx, 'lst_pKa'] = lst_pKa

        lst_df_result.append(df_results)

    df_pka = pd.concat(lst_df_result, ignore_index=True)
    df_pka.to_csv(path_data/'pKa_DMSO_1.csv', index=False)
    
    # pd.concat([pd.read_pickle(pkl)[1].assign(file=pkl.name) 
    #         for pkl in folder.glob('*_result.pkl')
    #         ], ignore_index=True)
    # path_old = Path(Path.home()/'old_pKalculator'/'pKalculator'/'src'/'pKalculator'/'submitit_pKalculator_1_alpb_DMSO')
    # df_results = load_pickles(folder=path_old)

    # for idx, row in df_results.iterrows():
    #     lst_pKa = calc_pKa_v2(df=df_results, row_idx=idx, pKa_ref=35, T=None)
    #     df_results.at[idx, 'lst_pKa'] = lst_pKa

    # print(df_results)

    print(df_pka)