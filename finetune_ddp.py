# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Multi-GPU (DistributedDataParallel) launcher for KERMT finetuning.

Mirrors pretrain_ddp.py: one process per GPU via torch.multiprocessing.spawn,
single-node NCCL rendezvous, data-parallel training with a DistributedSampler.
The finetune training/eval/checkpointing logic lives in task/train.py; this
file only bootstraps the process group and hands rank/world_size down through
cross_validate -> run_training.

Usage (data-parallel finetune across all visible GPUs):
    python finetune_ddp.py --data_path train.csv --separate_val_path val.csv \
        --separate_test_path test.csv --checkpoint_path <ckpt> --save_dir <dir> \
        --dataset_type regression --... (all normal finetune args)

Pin the GPU count explicitly with WORLD_SIZE (defaults to all visible GPUs):
    WORLD_SIZE=4 python finetune_ddp.py ...

Notes:
  - `args.batch_size` is the PER-GPU batch size; the effective global batch is
    batch_size * world_size (same convention as pretrain_ddp.py). The LR
    schedule is world-size-aware.
  - Single-process finetune (`python main.py finetune ...`) is unchanged and
    does not go through this launcher.
"""
import os
import sys

import torch
import torch.multiprocessing as mp
from rdkit import RDLogger
from torch.distributed import destroy_process_group

from kermt.data.torchvocab import MolVocab
from kermt.util.ddp_utils import configure_nccl_for_topology, ddp_setup
from kermt.util.parsing import parse_args
from kermt.util.utils import create_logger, setup_determinism
from task.cross_validate import cross_validate


def ddp_main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    # Suppress RDKit logging in every worker.
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    _ = MolVocab  # ensure vocab class is imported for checkpoint (un)pickling

    # Select the finetune subparser (mirror `main.py finetune ...`). Users invoke
    # this script with the normal finetune flags but no subcommand token.
    if len(sys.argv) < 2 or sys.argv[1] != 'finetune':
        sys.argv.insert(1, 'finetune')
    args = parse_args()

    # setup_determinism also sets CUBLAS_WORKSPACE_CONFIG, which
    # torch.use_deterministic_algorithms requires for cuBLAS on CUDA >= 10.2.
    setup_determinism(args.seed)

    # Only rank 0 writes logs; other ranks stay quiet.
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False) if rank == 0 else None

    cross_validate(args, logger, rank=rank, world_size=world_size)

    destroy_process_group()


if __name__ == '__main__':
    configure_nccl_for_topology()

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError('No GPUs found. DDP finetuning requires at least 1 GPU.')

    world_size = int(os.environ.get('WORLD_SIZE', available_gpus))
    if world_size > available_gpus:
        raise RuntimeError(
            f'WORLD_SIZE={world_size} but only {available_gpus} GPU(s) visible. '
            f'Set WORLD_SIZE<={available_gpus} or unset it to auto-detect.'
        )

    print(f'Launching DDP finetuning on {world_size}/{available_gpus} GPU(s)')
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size)
