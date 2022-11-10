import os 
import sys
os.chdir('../')
sys.path.append('src/pKalculator')
from core import calc_E_rel
from core import calc_pKa
from core import control_calcs
from solvents_parameters import gen_solvent_df
from solvents_parameters import find_all_combinations
import time
from rdkit import Chem
import pandas as pd

from pathlib import Path

root_path = Path.home()
print(root_path)
pKalculator_path = Path(root_path/'pKalculator')

df = pd.read_csv(Path(pKalculator_path/'test'/'compounds_paper.smiles'), sep=' ')


start = time.perf_counter()

# df = pd.read_csv(pKalculator_path/'test/compounds_paper.smiles', sep=' ')

print(df)

# test_name = 'furan#-1=2'

# print(df.loc[df['name'] == 'furan'])
# print(df.index[df['name'] == 'furan'])    

# # df['pKa_calc'] = None



# if test_name.split('#')[0] in df['name'].values:
#     print(df.loc[df['name'] == test_name.split('#')[0]])

# input_smiles = 'c1c(c2cc(sc2)C)n[nH]c1'
# input_mol = Chem.MolFromSmiles(input_smiles)
# name = 'test'

# best_conf_energy, best_conf_mol = calculateEnergy((input_mol, name))
# print(best_conf_energy)

# input_smiles = 'O1C=CC=C1'
# input_mol = Chem.MolFromSmiles(input_smiles)
# name = 'furan' 

# lst_name_deprot, lst_E_rel, lst_pKa_deprot = calc_E_rel(input_smiles, name, rxn_type='rm_proton')
# print(lst_name_deprot)
# print([name.split('#')[0] for name in lst_name_deprot])

# print(list(zip(lst_name_deprot, lst_E_rel, lst_pKa_deprot)))

df1 = df.loc[df['name'] == 'furan']
# df1 = df.loc[df['name'] == 'N,N-dimethyl-1H-pyrrol-1-amine']



xtb_params = {
            'method' : 2, 
            'solvent_model' : 'alpb',
            'solvent_name' : 'Ether'
            }

# from pathlib import Path

# home_path = Path.home()
# print(home_path)
# root = Path("/pKalculator/src/pKalculator/")
# print(f'root {root}')
# path = Path(home_path/'pKalculator/src/pKalculator/test_path')
# try:
#     path.mkdir(mode=0o755, parents=False, exist_ok=True)
# except FileExistsError:
#     print("Folder is already there")
# else:
#     print("Folder was created")

print(control_calcs(df=df1, xtb_params=xtb_params))




# print('--'*80)
# dict_iter_xtb = {
#             'method' : (1, 2), 
#             'solvent_model' : ('gbsa', 'alpb'),
#             'solvent_name' : None
#             }

# df_solvent = gen_solvent_df()
# df_iter_xtb = find_all_combinations(iter_dict=dict_iter_xtb, df_solvent=df_solvent)

# lst_all_dfs = []
# for idx, row in df_iter_xtb.iterrows():
    

#     xtb_params = {
#         'method' : row['method'], 
#         'solvent_model' : row['solvent_model'],
#         'solvent_name' : row['solvent_name']
#         }
#     # print(xtb_params)
#     df_control_calcs = control_calcs(df=df, xtb_params=xtb_params)
#     # print(df_control_calcs)
#     lst_all_dfs.append(df_control_calcs)

# df_all = pd.concat(lst_all_dfs, ignore_index=True)

# df_all.to_csv(Path(root_path/'pKalculator'/'test_calculations.csv'), index=False)



# print(df_all.head())



finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')
