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
The general utility functions.
"""
import csv
import logging
import os
import json
import pickle
import random
import math
from argparse import Namespace
from collections import defaultdict
from logging import Logger
from typing import List, Set, Tuple, Union, Dict
from collections import OrderedDict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn as nn
from tqdm import tqdm as core_tqdm

from kermt.data import MoleculeDatapoint, MoleculeDataset, StandardScaler
from kermt.model.models import KermtFpGeneration, KermtFinetuneTask
from kermt.util.nn_utils import initialize_weights
from kermt.util.scheduler import NoamLR


def seed_rngs(seed: int) -> None:
    """Seed the torch (CPU + CUDA), numpy, and python RNGs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_determinism(seed: int) -> None:
    """Seed all RNGs and enable deterministic algorithms.

    Shared by the finetune / HPO / DDP entry points (main.py, main_hpo.py,
    finetune_ddp.py) so their determinism setup stays in lockstep. Also sets
    CUBLAS_WORKSPACE_CONFIG, which torch.use_deterministic_algorithms requires
    for cuBLAS on CUDA >= 10.2.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_rngs(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def get_model_args():
    """
    Get model structure related parameters.
    
    Note: 'attn_out' is intentionally excluded. It controls the self-attention readout
    output size (FFN input = hidden_size * attn_out) and is only used during finetuning.
    Old checkpoints may have attn_out=128 saved, which causes model size to blow up.
    By excluding it, we allow the command-line/default value to take precedence.

    :return: a list containing parameters
    """
    # Changes: removed 'fine_tune_coff' from the list.
    return ['model_type', 'ensemble_size', 'input_layer', 'hidden_size', 'bias', 'depth',
            'dropout', 'activation', 'undirected', 'ffn_hidden_size', 'ffn_num_layers',
            'atom_message', 'weight_decay', 'select_by_loss', 'skip_epoch', 'backbone',
            'embedding_output_type', 'self_attention', 'attn_hidden', 'dense',
            'bond_drop_rate', 'distinct_init', 'aug_rate', 'nencoders',
            'dist_coff', 'no_attach_fea', 'coord', "num_attn_head", "num_mt_block",
            'num_tasks',  # Required for prediction on blinded data (no target columns)
            ]


def get_finetune_predict_consistency_args():
    """
    Get the arguments that should be consistent between finetune and predict modes.
    :return: a list containing parameters
    """
    return ['rdkit2D_normalization_type', 'features_generator']


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1



def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    datapoint_list = []
    for idx, datapoint in enumerate(data):
        if datapoint.smiles == '':
            print(f'invalid smiles {idx}: {datapoint.smiles}')
            continue
        mol = Chem.MolFromSmiles(datapoint.smiles)
        if mol is None:
            print(f'invalid smiles parse {idx}: {datapoint.smiles}')
            continue
        if mol.GetNumHeavyAtoms() == 0:
            print(f'invalid heavy {idx}')
            continue
        datapoint_list.append(datapoint)
    return MoleculeDataset(datapoint_list)


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
        args.features_dim = len(features_data[0])
    else:
        features_data = None
        if args is not None:
            args.features_dim = 0

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break
        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                args=args,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None,
                         args: Namespace = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint(line=[smile], args=args) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                # Default is JSON; fall back to .pkl only if the JSON variant
                # is not present (preserves old on-disk index directories).
                json_path = os.path.join(args.crossval_index_dir, f'{index}.json')
                pkl_path = os.path.join(args.crossval_index_dir, f'{index}.pkl')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as rf:
                        split_indices.extend(json.load(rf))
                else:
                    with open(pkl_path, 'rb') as rf:
                        split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        # Default is JSON. Explicit ``.pkl`` / ``.pckl`` opts into pickle (with a
        # python2 latin1 fallback for legacy binaries that may still be on disk).
        if folds_file.endswith(('.pkl', '.pckl')):
            try:
                with open(folds_file, 'rb') as f:
                    all_fold_indices = pickle.load(f)
            except UnicodeDecodeError:
                with open(folds_file, 'rb') as f:
                    all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        else:
            with open(folds_file, 'r') as f:
                all_fold_indices = json.load(f)
        # assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.smiles(), use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A Logger.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    # print some statistics about scaffolds
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float64)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels: {stats}')

    return stats


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)['args']



def get_ffn_layer_names(model: KermtFinetuneTask):
    """
    Get parameter tensor ids for task-specific layers in KermtFinetuneTask.
    
    Task-specific layers are those added for finetuning (not from pretrained checkpoint):
    - FFN layers: mol_atom_from_atom_ffn, mol_atom_from_bond_ffn
    - Readout attention layers: readout.attn (when using self_attention)
    
    These should be trained with full learning rate during finetuning.
    
    :param model: The KermtFinetuneTask model
    :return: List of parameter tensor memory ids for task-specific layers
    """
    # Readout and atom/bond ffn layers are returned
    return [name for name, _ in model.named_parameters() if "kermt" not in name]


def build_optimizer(model: nn.Module, args: Namespace):
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """

    # Only adjust the learning rate for the KermtFinetuneTask.
    if type(model) == KermtFinetuneTask:
        ffn_param_names = get_ffn_layer_names(model)
    else:
        # if not, init adam optimizer normally.
        return torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    base_params = [param for k, param in model.named_parameters() if k not in ffn_param_names]
    ffn_params = [param for k, param in model.named_parameters() if k in ffn_param_names]
    print(f"Number of base params: {len(base_params)}, number of ffn params: {len(ffn_params)}")
    if math.isclose(args.fine_tune_coff, 0.0, abs_tol=1e-6):
        print("Freezing parameters of encoder")
        for param in base_params:
            param.requires_grad = False
    else:
        print("Not freezing parameters of encoder")

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.init_lr * args.fine_tune_coff},
        {'params': ffn_params, 'lr': args.init_lr}
    ], lr=args.init_lr, weight_decay=args.weight_decay)

    return optimizer


def build_lr_scheduler(optimizer, args: Namespace, total_epochs: List[int] = None,
                       world_size: int = 1):
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be task.
    :param world_size: number of DDP processes. Under DDP each rank sees
        1/world_size of the data (via DistributedSampler), so the number of
        optimizer steps per epoch is divided by world_size (matches the effective
        global batch of batch_size * world_size, as in pretrain_ddp.py).
    :return: An initialized learning rate scheduler.
    """

    # Learning rate scheduler
    # When fine_tune_coff == 0, encoder is frozen and not in optimizer,
    # so we only have task params (1 group) with full LR (fine_tune_coff=1.0)
    # When fine_tune_coff > 0, we have 2 groups: encoder (index 0) and task (index 1)
    scheduler_fine_tune_coff = 1.0 if args.fine_tune_coff == 0 else args.fine_tune_coff

    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=args.train_data_size // (args.batch_size * world_size),
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        fine_tune_coff=scheduler_fine_tune_coff
    )


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None,
                    strict_shape_check: bool = False):
    """
    Loads a model checkpoint. This function can change the values of current_args according to args present in the checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :param strict_shape_check: If True, raise ValueError when a loaded parameter's shape
        does not match the current model. If False (default, required for CMIM-style
        workflows where pretrain checkpoints may have extra/partial params), log and
        skip the mismatched parameter.
    :return: The loaded MPNN and loaded checkpoint state
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    # TODO(sveccham): Change this to weights_only=True
    state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    args, loaded_state_dict = state['args'], state['state_dict']

    # Transform parameter names for compatibility
    # 1. Replace old "grover" naming with "kermt"
    loaded_state_dict = OrderedDict([(k.replace("grover", "kermt"), v) for k, v in loaded_state_dict.items()])

    # 2. Handle CMIM checkpoint format: strip "latent_dist." prefix
    #    This transforms "latent_dist.kermt.*" -> "kermt.*" for finetuning compatibility
    has_cmim_params = any("latent_dist." in k for k in loaded_state_dict.keys())
    if has_cmim_params:
        debug("Detected CMIM checkpoint format. Transforming 'latent_dist.kermt.*' -> 'kermt.*' for finetuning compatibility.")
    loaded_state_dict = OrderedDict([(k.replace("latent_dist.", ""), v) for k, v in loaded_state_dict.items()])

    model_ralated_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args

    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = build_model(current_args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            if strict_shape_check:
                raise ValueError(
                    f'Pretrained parameter "{param_name}" of shape {loaded_state_dict[param_name].shape} '
                    f'does not match corresponding model parameter of shape {model_state_dict[new_param_name].shape}.'
                )
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[new_param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model, state


def load_checkpoint_for_prediction(path: str,
                                   current_args: Namespace,
                                   cuda: bool = None,
                                   logger: logging.Logger = None):
    """
    Loads a finetuned model checkpoint for prediction.

    Unlike :func:`load_checkpoint` (which is permissive because CMIM-style
    pretrain checkpoints may have extra/partial params), this function enforces
    strict consistency: every loaded parameter must be present in the current
    model with the same shape, and arguments such as ``features_generator`` /
    ``rdkit2D_normalization_type`` must match those baked into the checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the model-related arguments
        loaded from the checkpoint.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MPNN.
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    # TODO(sveccham): Change this to weights_only=True
    state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    loaded_args, loaded_state_dict = state['args'], state['state_dict']

    loaded_state_dict = OrderedDict([(k.replace("grover", "kermt"), v) for k, v in loaded_state_dict.items()])

    # Check for consistency between finetune and predict arguments
    # ensuring that the model is used exactly how it was finetuned.
    finetune_predict_consistency_args = get_finetune_predict_consistency_args()
    for key, value in vars(current_args).items():
        if key in finetune_predict_consistency_args:
            if value != vars(loaded_args)[key]:
                raise ValueError(f'Argument {key} is not consistent between finetune and predict. '
                                 f'Finetune value: {value}, Predict value: {vars(loaded_args)[key]}')

    model_related_args = get_model_args()
    for key, value in vars(loaded_args).items():
        if key in model_related_args:
            setattr(current_args, key, value)

    # Build model
    model = build_model(current_args)
    model_state_dict = model.state_dict()

    # Load finetuned weights
    # Do not skip missing parameters
    # All parameters should have consistent size
    finetuned_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            raise ValueError(f'Finetuned parameter "{param_name}" cannot be found in model parameters.')
        if model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            raise ValueError(
                f'Finetuned parameter "{param_name}" of shape {loaded_state_dict[param_name].shape} '
                f'does not match corresponding model parameter of shape {model_state_dict[new_param_name].shape}.'
            )
        debug(f'Loading finetuned parameter "{param_name}".')
        finetuned_state_dict[new_param_name] = loaded_state_dict[param_name]

    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def get_loss_func(args: Namespace, model=None):
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if hasattr(model, "get_loss_func"):
        return model.get_loss_func(args)
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


def load_scalars(path: str):
    """
    Loads the scalars a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    # TODO(sveccham): Change this to weights_only=True
    state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def save_checkpoint(path: str,
                    model,
                    scaler,
                    features_scaler,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MPNN.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_model_for_restart(path:str, model, optimizer, scheduler, scaler, features_scaler, args, epoch):
    """
    Save the model, optimizer, and scheduler for restart. Saves model state_dict, optimizer state_dict, scheduler state_dict, data_scaler, features_scaler, and args.
    """
    # checkpoint_path is the path for pretrained model. It is not needed to restart finetuning.
    if hasattr(args, 'checkpoint_path'):
        delattr(args, 'checkpoint_path')
    state = {
        'args': args,
        'epoch': epoch,
        'state_dict': model.state_dict(), # model state_dict
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def build_model(args: Namespace, model_idx=0):
    """
    Builds a MPNN, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MPNN containing the MPN encoder along with final linear layers with parameters initialized.
    """
    if hasattr(args, 'num_tasks'):
        args.output_size = args.num_tasks
    else:
        args.output_size = 1

    if args.parser_name == "fingerprint":
        model = KermtFpGeneration(args)
    else:
        # finetune and evaluation case.
        model = KermtFinetuneTask(args)
    all_param_names = [name for name, _ in model.named_parameters()]
    initialize_weights(model=model, model_idx=model_idx, init_param_names=all_param_names)
    return model

def get_memory_usage():
    """
    Gets the current memory utilization of the Python process.
    
    :return: A tuple containing:
        - Current memory usage in MB
        - Peak memory usage in MB 
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Get current memory usage in bytes and convert to MB
    current_mem = process.memory_info().rss / 1024 / 1024
    
    # Get peak memory usage in bytes and convert to MB  
    peak_mem = process.memory_info().peak_wset / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wset') else 0
    
    return current_mem, peak_mem
