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
"""
Tests for vocab-label / graph-batch alignment.

`bond_random_mask` re-parses the SMILES independently of `MolGraph`, which *skips* bonds
dropped by `--bond_drop_rate`. Labels are matched to bonds by position alone, so the two
have to agree row for row. They are kept in step by gathering the labels through
`BatchMolGraph.rdkit_bond_idx` -- the same packed-row -> RDKit-bond-index map the
cuik-molmaker path already uses to reorder bond features.

These tests pin the two properties that matter: the gather is the identity when nothing
is dropped (so existing runs are unaffected), and it stays aligned when bonds are.
"""
from argparse import Namespace

import numpy as np

from kermt.data.kermtdataset import bond_random_mask
from kermt.data.molgraph import mol2graph

# Includes rings, where the packed bond order is a permutation of the RDKit bond order --
# the case a naive positional gather would get wrong.
SMILES = ['CCO', 'c1ccccc1N', 'CC(=O)Oc1ccccc1C(=O)O']


class FakeVocab:
    """Labels every bond with its own vocab id, so a shifted label is visible."""
    other_index = 1

    def __init__(self):
        self.stoi = _AlwaysHit()


class _AlwaysHit(dict):
    def get(self, key, default):  # noqa: D102 - mimics MolVocab.stoi.get
        return abs(hash(key)) % 500 + 2


def make_args(**overrides):
    args = Namespace(bond_drop_rate=0, no_cache=True,
                     use_cuikmolmaker_featurization=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def n_bond_rows(components):
    return components[1].shape[0]


def test_gather_is_identity_without_bond_dropout():
    """No dropout: the gathered labels must equal the legacy sequential ones exactly."""
    args = make_args()
    batch = mol2graph(list(SMILES), {}, args)

    np.random.seed(0)
    legacy = bond_random_mask(SMILES, FakeVocab())
    np.random.seed(0)
    gathered = bond_random_mask(SMILES, FakeVocab(), rdkit_bond_idx=batch.rdkit_bond_idx)

    assert gathered == legacy


def test_labels_match_bond_rows_with_bond_dropout():
    """With dropout the label vector must shrink to match the surviving bonds."""
    np.random.seed(0)
    args = make_args(bond_drop_rate=0.5)
    batch = mol2graph(list(SMILES), {}, args)
    components = batch.get_components()

    labels = bond_random_mask(SMILES, FakeVocab(), rdkit_bond_idx=batch.rdkit_bond_idx)

    # BondVocabPrediction emits one row per surviving undirected bond, plus padding.
    assert len(labels) == (n_bond_rows(components) - 1) // 2 + 1
    # Dropout must actually have dropped something, or the test proves nothing.
    assert len(labels) < len(bond_random_mask(SMILES, FakeVocab()))


def test_dropped_bonds_do_not_shift_surviving_labels():
    """Each surviving row keeps the label of the bond it actually holds."""
    np.random.seed(0)
    args = make_args(bond_drop_rate=0.5)
    batch = mol2graph(list(SMILES), {}, args)

    np.random.seed(1)
    labels = bond_random_mask(SMILES, FakeVocab(), rdkit_bond_idx=batch.rdkit_bond_idx)
    np.random.seed(1)
    by_rdkit_idx = bond_random_mask(SMILES, FakeVocab())  # legacy order == RDKit visit order

    # Rebuild the expected label for every surviving bond from the undropped run and
    # check it landed on the right row.
    full_batch = mol2graph(list(SMILES), {}, make_args())
    rdkit_to_label = {}
    for position, gidx in enumerate(full_batch.rdkit_bond_idx[0::2]):
        rdkit_to_label[gidx] = by_rdkit_idx[position + 1]

    for row, gidx in enumerate(batch.rdkit_bond_idx[0::2]):
        assert labels[row + 1] == rdkit_to_label[gidx]
