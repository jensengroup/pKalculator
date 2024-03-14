# pKalculator
pKalculator is a fully automated quantum chemistry (QM)-based workflow that computes the C-H pKa values of molecules. The QM workflow uses GNF2-xTB with ORCA on top.
pKalculator also includes an atom-based machine learning model (ML) to predict the C-H pKa values. The ML model (LightGBM regression model) is based on CM5 atomic charges that are computed using semiempirical tight binding (GFN1-xTB). 

For more, see [pKalculator: A pKa predictor for C-H bonds](https://www.google.com)

## Installation
We recommend using `conda` to get the required dependencies

    conda env create -f environment.yml && conda activate pkalculator

Download the latest version of xtb (v. 6.7.0)

    cd dep; wget https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz; tar -xvf ./xtb-6.7.0-linux-x86_64.tar.xz; cd ..


Hereafter, ORCA (v. 5.0.4) is required. Installation instructions can be found at https://sites.google.com/site/orcainputlibrary/setting-up-orca

ORCA requires a specific path for our QM workflow to work. Therefore, follow the comments in "src/qm_pkalculator/run_orca.py" and modify the path accordingly.

## Usage
Both our QM workflow and ML workflow are accessible through the command line in the terminal.

### QM workflow
Below is an example of how to start the QM workflow:

    python src/pkalculator/qm_pkalculator.py -cpus 5 -mem 10 -csv test.smiles -calc calc_test -submit submit_test -f CAM-B3LYP -b def2-TZVPPD -s DMSO -d -o -f

The arguments for the QM workflow are explained below:
| Arguments    | Description | 
| :------- |:---------|
| `-cpus` | Number of cpus per job. Defaults to 4 cpus |
| `-mem` | Amount of memory per job in GB. Defaults to 8 GB |
| `-p` | Set the SLURM partion to be used at your HPC. Defaults to kemi1 |
| `-csv` | csv path. The csv file must be comma seperated and contain a 'names' column and a 'smiles' column. |
| `-calc` | path for saving calculations. Defaults to 'calculations' |
| `-submit` | path for saving results from submitit. Defaults to 'submitit' |
| `-f` | The functional to be used. Defaults to 'CAM-B3LYP' |
| `-b` | which basis set. Defaults to 'def2-TZVPPD' |
| `-s` | solvent for the. Defaults to 'DMSO' |
| `-d` | set if D4 dispersion correction. |
| `-o` | set if optimization is needed. |
| `-q` | set if frequency computations are required. |


If needed, SLURM commands can be updated to work at your HPC.

- timeout_min| The total time that is allowed for each SLURM job before time out.
- slurm_array_parallelism| Maximum number SLURM jobs to run simultaneously.


### ML workflow
Below is an example of how to use the ML workflow:
    
    python src/pkalculator/ml_pkalculator.py -s CC(=O)Cc1ccccc1 -n comp2 -m ml_data/models/full_models/reg_model_all_data_dart.txt

The arguments for the ML workflow are explained below:
| Arguments    | Description | 
| :--- |:---------|
| `-s` | SMILES string. Defaults to 'CC(=O)Cc1ccccc1' |
| `-n` | Name of the compound. Defaults to 'comp2' |
| `-m` | Which model to be used. Defaults to the full regression model |
| `-e` | Identify the possible site of reaction within (e) pKa units of the lowest pKa value. Defaults to the full regression model. |

Hereafter, a list of tuples are returned [(0, 23.14), (3, 18.78), (5, 42.42), (6, 42.9), (7, 43.27)]. The first element in each tuple is the atom index and the second element in each tuple is the ML predicted pKa value for that atom index.

The workflow then produces an .png or .svg image of the molecule with its atom indices for easy comparison. The image of the molecule will also contain a teal circle that highlights the site with the lowest pKa value or within (e) pKa units from the lowest pKa value.

### Data
All additionl data can be found on ['https://sid.erda.dk/sharelink/EyuyjllJdp'](https://sid.erda.dk/sharelink/EyuyjllJdp)

Here the data is split into two folders `qm_data` and `ml_data` that represents QM workflow and the ML workflow, respectively. 

| Folder    | Description |
| :------- |:---------|
| `datasets` | Includes all datasets. Each `.pkl` contains a pandas DataFrame that can be loaded using the following command `pd.read_pickle(datasets/{dataset name}, compression={'method': 'gzip'})`. |
| `qm_data/calculations` | Includes all QM calculations, including .xyz files and .log files. |
| `ml_data/models` | Includes trained ML models on either all data or ML models trained on the training set (80 % of the data). |
| `ml_data/validation` | Includes data from the cross-validation for the ML models. The .log files gives an overview of the performance metrics. |

## Citation
