from rdkit import Chem
import time
from core import calculateEnergy
from gen_altered_smiles import change_mol_v6

def main():

    input_smiles = 'O1C=CC=C1'
    input_mol = Chem.MolFromSmiles(input_smiles)
    name = 'furan' 
    # input_mol = Chem.MolFromSmiles('c1c(c2cc(sc2)C)n[nH]c1')
    # input_mol = Chem.MolFromSmiles('c1cc(oc1)CCCN1C(=O)C[C@@H](C1=O)O')
    # input_mol = Chem.MolFromSmiles('c1ccccc1O')


    best_conf_energy_neutral, _ = calculateEnergy((input_mol, name))
    print(f'E_neutral = {best_conf_energy_neutral}')


    lst_name_deprot, _, lst_smiles_deprot, _, _ = change_mol_v6(name=name, smiles=input_smiles, rxn='rm_proton', reduce_smiles=True, show_mol=False)
    lst_mol_deprot = [Chem.MolFromSmiles(smi) for smi in lst_smiles_deprot]

    for name, mol in list(zip(lst_name_deprot, lst_mol_deprot)):
        best_conf_energy_deprot, _ = calculateEnergy((mol, name))
        print(name)
        print(f'E_deptrot = {best_conf_energy_deprot}')
        rel_E = best_conf_energy_deprot - best_conf_energy_neutral
        print(f'd_G = {rel_E:.2f} kcal/mol')

    

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')