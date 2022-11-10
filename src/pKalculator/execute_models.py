import os
from solvents_parameters import gen_solvent_df
from solvents_parameters import find_all_combinations
import subprocess
from pathlib import Path

submit_path = Path(Path.home()/'pKalculator'/'src'/'pKalculator')

dict_iter_xtb = {
        'method' : (1, 2), 
        'solvent_model' : ('gbsa', 'alpb'),
        'solvent_name' : None
        }

df_solvent = gen_solvent_df()
df_iter_xtb = find_all_combinations(iter_dict=dict_iter_xtb, df_solvent=df_solvent)
# df_iter_xtb = df_iter_xtb.head(3)
# print(df_iter_xtb)

for idx, row in df_iter_xtb.iterrows():
        method = row['method']
        solvent_model = row['solvent_model']
        solvent_name = row['solvent_name']
        print(method, solvent_model, solvent_name)

        #cmd = "python core.py 2 'alpb' 'DMSO'"
        s_path = submit_path/f"submitit_pKalculator{idx}"
        cmd = f"python core.py {method} {solvent_model} {solvent_name} {s_path}"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(p.communicate())
        p.communicate()

# subprocess.call("core.py 2 'alpb' 'DMSO'", shell=True)
# # os.system("core.py 2 'alpb' 'DMSO'")
