from pathlib import Path
import pandas as pd
import numpy as np
import itertools
from ase.units import Hartree, mol, kcal


path_data = Path(Path.home()/'pKalculator'/'data')
df_paper_neutral = pd.read_excel(path_data/"data_paper_neutral.ods", engine="odf")
df_paper_anions = pd.read_excel(path_data/"data_paper_anions.ods", engine="odf")


def calc_pKa_paper(df_neutral=df_paper_neutral, df_anion=df_paper_anions, ref_neutral='a1', ref_anion='a1-1', neutral=None, anion=None,  pKa_ref=35, T=None):
    """
    Het-H + Furan- --> Het- + Furan
    """
    if T == None:
        T = 298.15 #K
    
    gas_constant = 1.987 #kcal/(mol*K) or 8.31441 J/(mol*K) gas constant
    conversion_factor = Hartree * mol/kcal #convert energy from Hartree to kcal/mol
    
    idx_neutral_ref = df_neutral.index[df_neutral['Neutral'] == ref_neutral][0]
    idx_anion_ref = df_anion.index[df_anion['Anions'] == ref_anion][0]

    G_neutral_ref = df_neutral.at[idx_neutral_ref, 'G(gas)=TCG+E'] * conversion_factor + df_neutral.at[idx_neutral_ref, 'dG(sol)']
    G_anion_ref = df_anion.at[idx_anion_ref, 'G(gas)=TCG+E'] * conversion_factor + df_anion.at[idx_anion_ref, 'dG(sol)']

    idx_neutral = df_neutral.index[df_neutral['Neutral'] == neutral][0]
    idx_anion = df_anion.index[df_anion['Anions'] == anion][0]

    G_neutral = df_neutral.at[idx_neutral, 'G(gas)=TCG+E'] * conversion_factor + df_neutral.at[idx_neutral, 'dG(sol)']
    G_anion = df_anion.at[idx_anion, 'G(gas)=TCG+E'] * conversion_factor + df_anion.at[idx_anion, 'dG(sol)']
    
    dG_exchange = G_neutral_ref + G_anion - G_anion_ref - G_neutral

    pKa_paper = round(pKa_ref + (dG_exchange/(gas_constant*T*np.log(10)/1000)), 1)

    return pKa_paper

def get_pKa_from_entry(df_neutral=df_paper_neutral, df_anion=df_paper_anions, neutral=''):
    """
    Returns a tuple of anions and the pKa values calculated from the common neutral compound
    """
    idx_anions = df_anion.index[df_anion['Anions'].str.contains(neutral+'-', regex=False)].tolist()
    
    pKas = [calc_pKa_paper(df_neutral=df_paper_neutral, df_anion=df_paper_anions, ref_neutral='a1', ref_anion='a1-2', neutral=neutral, anion=df_anion.Anions[idx],  pKa_ref=35, T=None) for idx in idx_anions]
    anion_names = [df_anion.Anions[idx]for idx in idx_anions]
    tup_names_pKas =tuple(zip(anion_names, pKas))

    return tup_names_pKas

if __name__ == "__main__":
    import time

    start = time.perf_counter()

    paper_data = list(itertools.chain(*[get_pKa_from_entry(neutral=name) for name in df_paper_neutral.Neutral]))
    df_paper_data = pd.DataFrame(paper_data, columns=['name_paper', 'pKa_paper'])
    df_paper_data.to_csv(path_data/"pKa_paper.csv", index=False)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
