# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kermt.util.utils data-loading helpers (filter_invalid_smiles, split_data)."""

import pytest

from kermt.data import MoleculeDatapoint, MoleculeDataset
from kermt.util.utils import filter_invalid_smiles, split_data


def _dataset(smiles_list):
    """Build a MoleculeDataset from bare SMILES (no args -> no feature generation)."""
    return MoleculeDataset([MoleculeDatapoint(line=[smi]) for smi in smiles_list])


def test_filter_invalid_smiles_drops_unparseable_without_crashing():
    """Regression: SMILES that RDKit parses to None must be skipped, not crash.

    Before the fix, ``Chem.MolFromSmiles`` returning None led to
    ``None.GetNumHeavyAtoms()`` raising AttributeError. These inputs return
    None in modern RDKit: a syntax error ('C1CC1C(invalid') and a
    valence-invalid molecule ('C(C)(C)(C)(C)C', pentavalent carbon). The empty
    string is dropped by the pre-existing check.
    """
    data = _dataset(["c1ccccc1", "C1CC1C(invalid", "", "C(C)(C)(C)(C)C", "n1ccccc1"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["c1ccccc1", "n1ccccc1"]


def test_filter_invalid_smiles_passes_valid_through_unchanged():
    """All-valid input is returned unchanged and in order."""
    data = _dataset(["CCO", "c1ccccc1", "CC(=O)O"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["CCO", "c1ccccc1", "CC(=O)O"]


def test_filter_invalid_smiles_drops_zero_heavy_atom():
    """A SMILES that parses but has zero heavy atoms is filtered out."""
    data = _dataset(["[H][H]", "CCO"])
    filtered = filter_invalid_smiles(data)
    assert filtered.smiles() == ["CCO"]


_SPLIT_SMILES = ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CO", "CN"] * 10


@pytest.mark.parametrize("split_type", ["random", "scaffold_balanced"])
@pytest.mark.parametrize("sizes", [
    (0.8, 0.1, 0.1),
    (0.7, 0.2, 0.1),  # sums to 0.9999999999999999 in floating point
    (0.6, 0.3, 0.1),  # sums to 0.9999999999999999
    (0.7000000000000001, 0.2, 0.1),  # 1 - 0.2 - 0.1, as the skills compute train; sums to 1.0000000000000002
])
def test_split_data_accepts_sizes_with_float_rounding(split_type, sizes):
    """Split sizes that add up to 1 only up to float rounding must be accepted."""
    data = _dataset(_SPLIT_SMILES)
    train, val, test = split_data(data, split_type=split_type, sizes=sizes, seed=0)
    assert len(train) + len(val) + len(test) == len(data)


def test_split_data_rejects_sizes_not_summing_to_one():
    """Sizes that clearly do not add up to 1 are still rejected."""
    with pytest.raises(AssertionError):
        split_data(_dataset(_SPLIT_SMILES), split_type="random", sizes=(0.7, 0.2, 0.0), seed=0)
