import subprocess

import numpy as np
import torch
from rdkit import RDLogger

from kermt.util.parsing import parse_args, get_newest_train_args
from kermt.util.utils import create_logger, setup_determinism
from task.cross_validate import cross_validate
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction
from kermt.data.torchvocab import MolVocab


def get_git_branch_commit():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        return branch, commit
    except Exception:
        return None, None

class UserError(Exception):
    pass

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
