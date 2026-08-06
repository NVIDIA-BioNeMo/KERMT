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
The fingerprint generation function.
"""
import copy
from argparse import Namespace
from logging import Logger
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kermt.data import MolCollator
from kermt.data import MoleculeDataset
from kermt.util.utils import get_data, create_logger, load_checkpoint


def do_generate(model: nn.Module,
                data: MoleculeDataset,
                args: Namespace,
                ) -> List[List[float]]:
    """
    Do the fingerprint generation on a dataset using the pre-trained models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: A StandardScaler object fit on the training targets.
    :return: A list of fingerprints.
    """
    model.eval()
    # Disable bond dropout locally without mutating the shared args (same pattern
    # as predict(); see issue #22).
    args_copy = copy.copy(args)
    args_copy.bond_drop_rate = 0
    preds = []

    mol_collator = MolCollator(args=args_copy, shared_dict={})

    num_workers = 0
    mol_loader = DataLoader(data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=mol_collator)
    for item in mol_loader:
        _, batch, features_batch, _, _ = item
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            preds.extend(batch_preds.data.cpu().numpy())
    return preds


def generate_fingerprints(args: Namespace, logger: Logger = None) -> List[List[float]]:
    """
    Generate the fingerprints.

    :param logger:
    :param args: Arguments.
    :return: A list of lists of target fingerprints.
    """

    checkpoint_path = args.checkpoint_paths[0]
    if logger is None:
        logger = create_logger('fingerprints', quiet=False)
    print('Loading data')
    test_data = get_data(path=args.data_path,
                         args=args,
                         use_compound_names=False,
                         max_data_size=float("inf"),
                         skip_invalid_smiles=False)
    test_data = MoleculeDataset(test_data)

    logger.info(f'Total size = {len(test_data):,}')
    logger.info(f'Generating...')
    # Load model
    model, _ = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
    model_preds = do_generate(
        model=model,
        data=test_data,
        args=args
    )

    return model_preds
