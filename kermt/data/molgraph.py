# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The data structure of Molecules.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/features/featurization.py
"""
from argparse import Namespace
from typing import List, Tuple, Union

import numpy as np
import torch
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import cuik_molmaker
import torch
from cuik_molmaker.mol_features import MoleculeFeaturizer
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from kermt.util.features import FeatureRange, get_feature_range
from kermt.util._cuik_compat import (
    atom_onehot_feature_names_to_tensor,
    atom_float_feature_names_to_tensor,
    bond_feature_names_to_tensor,
    to_float_tensor,
)

# Atom feature sizes
MAX_ATOMIC_NUM = 100


ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM + 18


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding




class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str,  args: Namespace = None):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.args = args
        self.n_atoms = 0  # number of atoms 
        self.n_bonds = 0  # number of bonds (after considering bond drop rate)
        self.n_rdkit_bonds = 0 # number of bonds in rdkit molecule (without considering bond drop rate)
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.rdkit_bond_idx = [] # mapping from bond index to rdkit bond index

        # If use_cuikmolmaker_featurization is True, then compute_atom_bond_features is False
        self.compute_atom_bond_features = not args.use_cuikmolmaker_featurization

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # Check if SMILES is valid
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'. RDKit failed to parse it.")

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        self.n_rdkit_bonds = mol.GetNumBonds() # number of bonds in rdkit molecule (without considering bond drop rate)

        if self.compute_atom_bond_features:
            self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
            self.hydrogen_acceptor = Chem.MolFromSmarts(
                "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
                "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
            self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
            self.basic = Chem.MolFromSmarts(
                "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
                "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

            self.hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
            self.hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
            self.acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
            self.basic_match = sum(mol.GetSubstructMatches(self.basic), ())
            self.ring_info = mol.GetRingInfo()

            # Get atom features
            for _, atom in enumerate(mol.GetAtoms()):
                self.f_atoms.append(self.atom_features(atom))
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]


        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                if args.bond_drop_rate > 0:
                    if np.random.binomial(1, args.bond_drop_rate):
                        continue

                # If bond if dropped, we do not need to store it in rdkit bond index
                bidx = bond.GetIdx()
                self.rdkit_bond_idx.extend([bidx, bidx])

                if self.compute_atom_bond_features:
                    f_bond = self.bond_features(bond)
                    # Always treat the bond as directed.
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                # print(f"{a1=} {a2=} {self.a2b=}")
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

    def atom_features(self, atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for an atom.

        :param atom: An RDKit atom.
        :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
        :return: A list containing the atom features.
        """
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
                   onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
                   onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
                   onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
                   onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
                   onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
                   [1 if atom.GetIsAromatic() else 0] + \
                   [atom.GetMass() / 100]

        atom_idx = atom.GetIdx()
        features = features + \
                   onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                   [atom_idx in self.hydrogen_acceptor_match] + \
                   [atom_idx in self.hydrogen_donor_match] + \
                   [atom_idx in self.acidic_match] + \
                   [atom_idx in self.basic_match] + \
                   [self.ring_info.IsAtomInRingOfSize(atom_idx, 3),
                    self.ring_info.IsAtomInRingOfSize(atom_idx, 4),
                    self.ring_info.IsAtomInRingOfSize(atom_idx, 5),
                    self.ring_info.IsAtomInRingOfSize(atom_idx, 6),
                    self.ring_info.IsAtomInRingOfSize(atom_idx, 7),
                    self.ring_info.IsAtomInRingOfSize(atom_idx, 8)]
        return features

    def bond_features(self, bond: Chem.rdchem.Bond
                      ) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for a bond.

        :param bond: A RDKit bond.
        :return: A list containing the bond features.
        """

        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace, prepared_atom_feats=None, prepared_bond_feats=None):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + self.atom_fdim

        self.prepared_atom_feats = prepared_atom_feats # store raw features from cuik-molmaker
        self.prepared_bond_feats = prepared_bond_feats # store raw features from cuik-molmaker

        if args.use_cuikmolmaker_featurization and prepared_atom_feats is None:
            raise ValueError("If using cuik-molmaker, atoms features should be prepared before BatchMolGraph is initialized.")
        if args.use_cuikmolmaker_featurization and prepared_bond_feats is None:
            raise ValueError("If using cuik-molmaker, bond features should be prepared before BatchMolGraph is initialized.")

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_rdkit_bonds = 0 # # number of bonds in rdkit molecule (without considering bond drop rate)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        if not args.use_cuikmolmaker_featurization:
            f_atoms = [[0] * self.atom_fdim]  # atom features
            f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        rdkit_bond_idx = []
        b2revb = [0]  # mapping from bond index to the index of the reverse bond

        for mol_graph in mol_graphs:
            if not args.use_cuikmolmaker_featurization:
                f_atoms.extend(mol_graph.f_atoms)
                f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            rdkit_bond_idx.extend([bidx + self.n_rdkit_bonds for bidx in mol_graph.rdkit_bond_idx])
            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            self.n_rdkit_bonds += mol_graph.n_rdkit_bonds

        # Packed directed-bond row -> global RDKit bond index, two entries per surviving
        # bond. Already used below to reorder cuik-molmaker features; kept on the batch so
        # the vocab task can align its per-bond labels to the rows that actually survived
        # --bond_drop_rate. Note this indexes f_bonds from row 1, since row 0 is padding.
        self.rdkit_bond_idx = rdkit_bond_idx

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        if not args.use_cuikmolmaker_featurization:
            self.f_atoms = torch.FloatTensor(f_atoms)
        else:
            # Add empty rows
            zeros = torch.zeros(1, prepared_atom_feats.shape[1], dtype=prepared_atom_feats.dtype)
            self.f_atoms = torch.cat([zeros, prepared_atom_feats], dim=0)

        if not args.use_cuikmolmaker_featurization:
            self.f_bonds = torch.FloatTensor(f_bonds)
        else:
            # Prepare bond features for KERMT model
            # 1. Remove duplicate bonds features
            prepared_bond_feats_no_dup = prepared_bond_feats[torch.arange(0, prepared_bond_feats.shape[0], 2)]
            # 2. Arrange bond features in order of rdkit bond index
            prepared_bond_feats_scattered = prepared_bond_feats_no_dup[rdkit_bond_idx]
            # 3. Prepare atom features for each bond
            prepared_atom_feats_scattered = self.f_atoms[b2a, :]
            # 4. Add zeros row to the beginning of the bond features
            zeros = torch.zeros(1, prepared_bond_feats.shape[1], dtype=prepared_bond_feats.dtype)
            # 5. Concatenate zeros at the top of bond features
            prepared_bond_feats_scattered2 = torch.cat([zeros, prepared_bond_feats_scattered], dim=0)
            # 6. Concatenate atom features and bond features to get final bond features
            self.f_bonds = torch.cat([prepared_atom_feats_scattered, prepared_bond_feats_scattered2], dim=1)

        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = self.b2a[self.a2b]  # only needed if using atom messages
        self.a_scope = torch.LongTensor(self.a_scope)
        self.b_scope = torch.LongTensor(self.b_scope)

    def set_new_atom_feature(self, f_atoms):
        """
        Set the new atom feature. Do not update bond feature.
        :param f_atoms:
        """
        self.f_atoms = f_atoms

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.a2a

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str], shared_dict,
              args: Namespace, set_seed: bool = False, cmm_feature_range: dict[str, FeatureRange] = None, cmm_tensors: dict[str, torch.Tensor] = None) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :param set_seed: Whether to set the seed or not.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    if args.use_cuikmolmaker_featurization:
        if cmm_feature_range is None:
            raise ValueError("If using cuik-molmaker, cmm_feature_range must be provided.")
        if cmm_tensors is None:
            raise ValueError("If using cuik-molmaker, cmm_tensors must be provided.")
    
    # Set seed
    if set_seed:
        seed = getattr(args, "seed", None)
        if seed is not None:
            print(f"Setting seed to {seed}")
            np.random.seed(seed)
        else:
            raise ValueError(f"Seed is not set in args: {args=}")

    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in shared_dict:
            mol_graph = shared_dict[smiles]
        else:
            mol_graph = MolGraph(smiles, args=args)
            if not args.no_cache:
                shared_dict[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    if args.use_cuikmolmaker_featurization:


        atom_props_onehot_array = cmm_tensors["atom_onehot"]
        atom_props_float_array = cmm_tensors["atom_float"]
        bond_props_array = cmm_tensors["bond"]
        add_h, offset_carbon, duplicate_edges, add_self_loop = False, False, True, False
        batch_feats = cuik_molmaker.batch_mol_featurizer(smiles_batch, atom_props_onehot_array, atom_props_float_array, bond_props_array, add_h, offset_carbon, duplicate_edges, add_self_loop)
        atom_feats_cmm, bond_feats_cmm, _, _, _ = batch_feats
        atom_feats_cmm = to_float_tensor(atom_feats_cmm)
        bond_feats_cmm = to_float_tensor(bond_feats_cmm)

        # For atomic features, cuik-molmaker always returns one-hot encoded features first followed by float features
        # We need to rearrange the features to match the order of the features expected by KERMT model

        atom_feats_cmm = torch.cat([
            atom_feats_cmm[:, :cmm_feature_range["hybridization"].end],
            atom_feats_cmm[:, cmm_feature_range["aromatic"].start: cmm_feature_range["mass"].end],
            atom_feats_cmm[:, cmm_feature_range["implicit-valence"].start:cmm_feature_range["implicit-valence"].end],
            atom_feats_cmm[:, cmm_feature_range["hydrogen-bond-acceptor"].start:],
            atom_feats_cmm[:, cmm_feature_range["ring-size"].start:cmm_feature_range["ring-size"].end],
        ], axis=1)
        # Reorder formal charge
        atom_feats_cmm[:, cmm_feature_range["formal-charge"].start:cmm_feature_range["formal-charge"].end] = atom_feats_cmm[:, [112, 110, 111, 113, 109, 108]]

        # Round mass idx to 6 decimal places
        # Somehow the model is pretty sensitive to number of decimal places
        # New mass feature is at index 132
        atom_feats_cmm[:, 132] = torch.round(atom_feats_cmm[:, 132], decimals=6)
    else:
        atom_feats_cmm = None
        bond_feats_cmm = None

    return BatchMolGraph(mol_graphs, args, prepared_atom_feats=atom_feats_cmm, prepared_bond_feats=bond_feats_cmm)


class MolCollator(object):
    """
    Collator for pytorch dataloader
    :param shared_dict: a shared dict of multiprocess.
    :param args: Arguments.
    """
    def __init__(self, shared_dict, args):
        self.args = args
        self.shared_dict = shared_dict

        if not hasattr(args, "features_generator"):
            # This case is needed for fingerprint mode 
            self.rdkit2d_featurizer = None
        elif args.features_generator is None:
            self.rdkit2d_featurizer = None
        elif "rdkit_2d_normalized_cuik_molmaker" in args.features_generator:
            # Get reference list of RDKit 2D descriptors from descriptastorus
            desc_ref = [x[0] for x in rdDescriptors.RDKit2D().columns]
            self.rdkit2d_featurizer = MoleculeFeaturizer(molecular_descriptor_type="rdkit2D", rdkit2D_descriptor_list = desc_ref, rdkit2D_normalization_type=args.rdkit2D_normalization_type)
        else:
            self.rdkit2d_featurizer = None

        if args.use_cuikmolmaker_featurization:
            # Form feature arrays for cuik-molmaker
            self.cmm_feature_arrays = {}
            atom_onehot_props = ["atomic-number", "total-degree", "formal-charge", "chirality",
                                "num-hydrogens", "hybridization",
                                "implicit-valence", 
                                "ring-size",
                                ]

            self.cmm_feature_arrays["atom_onehot"] = atom_onehot_feature_names_to_tensor(atom_onehot_props)
            atom_float_props = ["aromatic", "mass",
                                "hydrogen-bond-acceptor",
                                    "hydrogen-bond-donor",
                                    "acidic", "basic"
                                    ]
            self.cmm_feature_arrays["atom_float"] = atom_float_feature_names_to_tensor(atom_float_props)
            bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
            self.cmm_feature_arrays["bond"] = bond_feature_names_to_tensor(bond_props)

            # Get feature ranges for cuik-molmaker
            self.cmm_feature_range = get_feature_range(atom_onehot_props, atom_float_props)

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        # Generate features_batch
        # Check if there are features that should be generated on the fly.
        ## Only rdkit_2d_normalized_onthefly and rdkit_2d_normalized_cuik_molmaker are supported
        ## for on the fly featurization.

        if batch[0].features_generator is None:
            features_batch = [d.features for d in batch]
        elif "rdkit_2d_normalized_onthefly" in batch[0].features_generator:
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
            features_batch = []
            for smiles in smiles_batch:
                features = generator.process(smiles)[1:]
                features_batch.append(features)
        elif "rdkit_2d_normalized_cuik_molmaker" in batch[0].features_generator:
            features_batch = self.rdkit2d_featurizer.featurize(smiles_batch)
        else:
            features_batch = [d.features for d in batch]

        target_batch = [d.targets for d in batch]
        if self.args.use_cuikmolmaker_featurization:
            batch_mol_graph = mol2graph(smiles_batch, self.shared_dict, self.args, cmm_feature_range=self.cmm_feature_range, cmm_tensors=self.cmm_feature_arrays)
        else:
            batch_mol_graph = mol2graph(smiles_batch, self.shared_dict, self.args)
        batch = batch_mol_graph.get_components()

        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        return smiles_batch, batch, features_batch, mask, targets
