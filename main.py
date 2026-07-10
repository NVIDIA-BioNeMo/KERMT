import os
import subprocess

import numpy as np
import torch
import torch.multiprocessing as mp
from rdkit import RDLogger
from torch.distributed import destroy_process_group

from kermt.data.torchvocab import MolVocab
from kermt.util.ddp_utils import configure_nccl_for_topology, ddp_setup
from kermt.util.parsing import parse_args, get_newest_train_args
from kermt.util.utils import create_logger, setup_determinism
from task.cross_validate import cross_validate
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction


def get_git_branch_commit():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        return branch, commit
    except Exception:
        return None, None

class UserError(Exception):
    pass


def _finetune_ddp_worker(rank, world_size, args):
    """One data-parallel finetune process, spawned per GPU by the finetune
    entrypoint. Mirrors the single-process finetune path but pins the process to
    ``rank`` and runs cross_validate with the DDP ``world_size``.
    """
    ddp_setup(rank, world_size)
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    _ = MolVocab  # keep vocab class imported for checkpoint (un)pickling in every worker
    setup_determinism(args.seed)
    # Only rank 0 writes logs; other ranks stay quiet.
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False) if rank == 0 else None
    cross_validate(args, logger, rank=rank, world_size=world_size)
    destroy_process_group()


if __name__ == '__main__':
    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args()

    # setup random seed
    print(f"Setting up with random seed: {args.seed}")
    setup_determinism(args.seed)
    print(f"args: {args}")

    branch, commit = get_git_branch_commit()
    print(f"Git branch: {branch}, Commit: {commit}")
    
    if args.parser_name == 'finetune':
        # Single entrypoint for both single-process and multi-GPU (DDP) finetune.
        # If WORLD_SIZE is set in the environment, run data-parallel across that many
        # GPUs (one spawned process per GPU); otherwise run the single-process path.
        # WORLD_SIZE=1 exercises the DDP path on a single GPU. --batch_size is per-GPU
        # under DDP, matching pretrain_ddp.py.
        ws_env = os.environ.get('WORLD_SIZE')
        if ws_env is not None:
            world_size = int(ws_env)
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                raise UserError('No GPUs found; DDP finetuning requires at least 1 GPU.')
            if world_size > available_gpus:
                raise UserError(
                    f'WORLD_SIZE={world_size} but only {available_gpus} GPU(s) visible. '
                    f'Set WORLD_SIZE<={available_gpus} or unset it.'
                )
            configure_nccl_for_topology()
            print(f'Launching DDP finetuning on {world_size}/{available_gpus} GPU(s)')
            mp.spawn(_finetune_ddp_worker, args=(world_size, args), nprocs=world_size)
        else:
            logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
            cross_validate(args, logger)
    elif args.parser_name == 'pretrain':
        logger = create_logger(name='pretrain', save_dir=args.save_dir)
        raise UserError("Run pretraining using pretrain_ddp.py")
    elif args.parser_name == "eval":
        logger = create_logger(name='eval', save_dir=args.save_dir, quiet=False)
        cross_validate(args, logger)
    elif args.parser_name == 'fingerprint':
        train_args = get_newest_train_args()
        logger = create_logger(name='fingerprint', save_dir=None, quiet=False)
        feas = generate_fingerprints(args, logger)
        np.savez_compressed(args.output_path, fps=feas)
    elif args.parser_name == 'predict':
        train_args = get_newest_train_args()
        avg_preds, test_smiles = make_predictions(args, train_args)
        write_prediction(avg_preds, test_smiles, args)
