import os 
import sys
os.chdir('../')
sys.path.append('src/pKalculator')
from core import calc_E_rel
from core import calc_pKa
import time
from rdkit import Chem
import pandas as pd


start = time.perf_counter()

df = pd.read_csv('test/compounds_paper.smiles', sep=' ')#.set_index('name')

print(df)

test_name = 'furan#-1=2'

print(df.loc[df['name'] == 'furan'])
print(df.index[df['name'] == 'furan'])

df['pKa_calc'] = None



if test_name.split('#')[0] in df['name'].values:
    print(df.loc[df['name'] == test_name.split('#')[0]])

# input_smiles = 'c1c(c2cc(sc2)C)n[nH]c1'
# input_mol = Chem.MolFromSmiles(input_smiles)
# name = 'test'

# best_conf_energy, best_conf_mol = calculateEnergy((input_mol, name))
# print(best_conf_energy)

input_smiles = 'O1C=CC=C1'
input_mol = Chem.MolFromSmiles(input_smiles)
name = 'furan' 

lst_name_deprot, lst_E_rel, lst_pKa_deprot = calc_E_rel(input_smiles, name, rxn_type='rm_proton')
print(lst_name_deprot)
print([name.split('#')[0] for name in lst_name_deprot])

print(list(zip(lst_name_deprot, lst_E_rel, lst_pKa_deprot)))



#     if name_deprot.split('#')[0] in df['name'].values:
#         idx = df.index[df['name'] == name_deprot.split('#')[0]]
#         df.loc[idx, 'New Column Title'] = "some value"

#     print(name_deprot)



finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')
