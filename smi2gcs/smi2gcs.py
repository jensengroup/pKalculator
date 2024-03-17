# MIT License
#
# Copyright (c) 2022 Nicolai Ree
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import argparse
from rdkit import Chem
from DescriptorCreator.PrepAndCalcDescriptor import Generator


def parse_args():
    """
    Argument parser so this can be run from the command line
    """
    parser = argparse.ArgumentParser(
        description="Generate atomic descriptors from the command line"
    )
    parser.add_argument(
        "-s",
        "--smiles",
        default="c1(ccno1)C",
        help="SMILES representation of the molecule for which the atomic descriptors are to be generated",
    )
    parser.add_argument(
        "-a",
        "--atom_sites",
        default="0,1",
        help="A list of atom indices for which atomic descriptors are to be generated",
    )
    parser.add_argument(
        "-n", "--name", default="test_mol", help="The name of the molecule"
    )
    return parser.parse_args()


# For command line use
if __name__ == "__main__":
    args = parse_args()

    generator = Generator()
    des = (
        "GraphChargeShell",
        {"charge_type": "cm5", "n_shells": 6, "use_cip_sort": True},
    )

    # smiles = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles), isomericSmiles=True) # canonicalize input smiles
    smiles = args.smiles
    atom_sites = [int(i) for i in args.atom_sites.split(",")]

    cm5_list = generator.calc_CM5_charges(
        smiles, name=args.name, optimize=False, save_output=True
    )
    atom_indices, descriptor_vector, mapper_vector = generator.create_descriptor_vector(
        atom_sites, des[0], **des[1]
    )

    print("SMILES:", smiles)
    print("Atom indices:", atom_indices)
    print("Atom descriptors", descriptor_vector)
    print("Atom mapper", mapper_vector)
