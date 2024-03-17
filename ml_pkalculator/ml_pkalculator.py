import sys
from pathlib import Path

home_directory = Path.cwd()
if home_directory.name != "pKalculator":
    raise ValueError("Please run this script from the pKalculator directory")
sys.path.append(str(home_directory / "qm_pkalculator"))
sys.path.append(str(home_directory / "smi2gcs"))

import argparse

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from io import BytesIO
from PIL import Image
from collections import defaultdict, OrderedDict
import lightgbm as lgb

from modify_smiles import deprotonate, remove_Hs
from DescriptorCreator.PrepAndCalcDescriptor import Generator


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Get ML predicted C-H pKa values of a molecule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--smiles",
        default="CC(=O)Cc1ccccc1",
        help="SMILES input for ML prediction of pKa values",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--name",
        default="comp2",
        help="The name of the molecule",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="model/reg_model_all_data_dart.txt",
        help="Path to the model file to use for prediction",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--error",
        default=0.0,
        help="Identify the possible site of reaction within (number) pKa units of the lowest pKa value",
        type=float,
    )

    args = parser.parse_args()

    return args


def draw_mol_highlight(
    smiles,
    lst_atomindex,
    lst_pka_pred,
    error=0.0,
    legend="",
    img_size=(350, 300),
    draw_option="png",
    draw_legend=False,
    save_folder="",
    name="test",
):

    rdDepictor.SetPreferCoordGen(True)
    highlightatoms = defaultdict(list)
    atomrads = {}
    dict_color = {
        "green": (0.2, 1, 0.0, 1),
        "teal": (0.0, 0.5, 0.5, 1),
    }

    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDepictor.StraightenDepiction(rdkit_mol)

    dict_atomidx_pka = {
        atom_index: pka
        for atom_index, pka in zip(lst_atomindex, lst_pka_pred)
        if abs(pka - min(lst_pka_pred)) <= error
    }

    sorted_dict_atomidx_pka = OrderedDict(
        sorted(dict_atomidx_pka.items(), key=lambda x: x[1], reverse=False)
    )

    for atom_idx, atom in enumerate(rdkit_mol.GetAtoms()):
        if atom_idx in sorted_dict_atomidx_pka.keys():
            highlightatoms[atom_idx].append(dict_color["teal"])
            atomrads[atom_idx] = 0.2
            label = f"{sorted_dict_atomidx_pka[atom_idx]:.2f}"
            # atom.SetProp("atomNote", label)

    if draw_option == "png":
        d2d = Draw.MolDraw2DCairo(img_size[0], img_size[1])
    elif draw_option == "svg":
        d2d = Draw.MolDraw2DSVG(img_size[0], img_size[1])
    dopts = d2d.drawOptions()
    dopts.addAtomIndices = True
    dopts.legendFontSize = 35  # legend font size
    dopts.atomHighlightsAreCircles = True
    dopts.fillHighlights = True
    dopts.annotationFontScale = 0.9
    dopts.centreMoleculesBeforeDrawing = True
    dopts.fixedScale = 0.95  # -1.0 #0.5
    # dopts.drawMolsSameScale = False
    mol = Draw.PrepareMolForDrawing(rdkit_mol)
    if draw_legend:
        d2d.DrawMoleculeWithHighlights(
            mol, legend, dict(highlightatoms), {}, dict(atomrads), {}
        )
    else:
        d2d.DrawMoleculeWithHighlights(
            mol, "", dict(highlightatoms), {}, dict(atomrads), {}
        )
    d2d.FinishDrawing()

    if draw_option == "png":
        bio = BytesIO(d2d.GetDrawingText())
        save_path = Path(f"{save_folder}/{name}.png")
        img = Image.open(bio)
        img.save(
            save_path,
            dpi=(700, 600),
            transparent=False,
            facecolor="white",
            format="PNG",
        )
        # return (legend.split(" ")[0], Image.open(bio))
    elif draw_option == "svg":
        svg = d2d.GetDrawingText()
        svg.replace("svg:", "")
        with open(f"{save_folder}/{name}.svg", "w") as f:
            f.write(svg)
        # return (legend.split(" ")[0], svg)


if __name__ == "__main__":
    args = get_args()
    smiles = args.smiles
    name = args.name
    model = args.model
    error = args.error
    model = Path.cwd() / model
    save_folder = Path.cwd() / "data/ml_predictions"
    if not save_folder.exists():
        save_folder.mkdir(exist_ok=True)

    if not model.exists():
        raise ValueError(
            f"Model file {model} does not exist. Check model path and try again"
        )
    reg_model_full = lgb.Booster(model_file=model)
    print("-" * 50)
    print(f"Loaded model: {model}")
    print("-" * 50)
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    (
        lst_atomindex_deprot,
        lst_atomsite_deprot,
        lst_smiles_deprot,
        lst_smiles_map_deprot,
        lst_mol_deprot,
        lst_names_deprot,
        dict_atom_idx_to_H_indices,
    ) = remove_Hs(
        name=name,
        smiles=smiles,
        rdkit_mol=None,
        atomsite=None,
        gen_all=True,
        remove_H=True,
    )

    print(f"found {len(lst_atomindex_deprot)} C-H sites: {lst_atomindex_deprot}")
    generator = Generator()
    des = (
        "GraphChargeShell",
        {"charge_type": "cm5", "n_shells": 6, "use_cip_sort": True},
    )

    try:
        cm5_list = generator.calc_CM5_charges(
            smi=smiles, name=name, optimize=False, save_output=True
        )
        (
            atom_indices,
            descriptor_vector,
            mapper_vector,
        ) = generator.create_descriptor_vector(lst_atomindex_deprot, des[0], **des[1])
    except Exception as e:
        print(e)
        print(f"Error in generating descriptor vector for {name}.")
        descriptor_vector = None

    ML_predicted_pKa_values = reg_model_full.predict(descriptor_vector)
    print("ML predicted C-H pKa values")
    print(
        f"{[(atom_idx, round(pka, 2)) for atom_idx, pka in zip(atom_indices, ML_predicted_pKa_values)]}"
    )

    draw_mol_highlight(
        smiles=smiles,
        lst_atomindex=lst_atomindex_deprot,
        lst_pka_pred=ML_predicted_pKa_values,
        error=error,
        legend="",
        img_size=(350, 300),
        draw_option="svg",
        draw_legend=False,
        save_folder=save_folder,
        name=name,
    )
