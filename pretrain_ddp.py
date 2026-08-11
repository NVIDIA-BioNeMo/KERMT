# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import torch
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
import os
import random
import signal
import threading

import numpy as np

# ============================================================================
# Graceful Shutdown Handling
# ============================================================================
# This allows the training to checkpoint and exit cleanly when the cluster
# sends a signal (e.g., SIGUSR1 before time limit, or SIGTERM).
# This is shared across all spawned processes via multiprocessing.Value.

# Global shutdown event (used within each process)
_SHUTDOWN_REQUESTED = threading.Event()

def _graceful_shutdown_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"[SIGNAL] Received {signal_name}. Requesting graceful shutdown...", flush=True)
    _SHUTDOWN_REQUESTED.set()

def setup_signal_handlers():
    """
    Set up signal handlers for graceful shutdown.
    
    SIGUSR1: Commonly used by SLURM to warn before time limit
    SIGTERM: Standard termination signal
    SIGINT: Ctrl+C on workstations
    
    Call this in each spawned process if you want graceful shutdown.
    """
    # SIGUSR1 is commonly used by SLURM for pre-termination warning
    signal.signal(signal.SIGUSR1, _graceful_shutdown_handler)
    # SIGTERM is the standard termination signal
    signal.signal(signal.SIGTERM, _graceful_shutdown_handler)
    # SIGINT is Ctrl+C - useful for graceful shutdown on workstations
    signal.signal(signal.SIGINT, _graceful_shutdown_handler)
    print("[INFO] Signal handlers set up for graceful shutdown (SIGUSR1, SIGTERM, SIGINT)", flush=True)

def is_shutdown_requested():
    """Check if graceful shutdown has been requested."""
    return _SHUTDOWN_REQUESTED.is_set()


class ResumableDistributedSampler(DistributedSampler):
    """
    A DistributedSampler that supports resuming from a specific batch index.
    
    This is crucial for large datasets where we can only complete a fraction of an epoch
    per job. Instead of loading and skipping batches (slow), we skip at the index level
    (fast - no data loading needed for skipped samples).
    
    Key features:
    - Deterministic shuffle when set_epoch() is called (same as parent class)
    - Can skip the first N samples efficiently
    - Properly handles the reduced dataset length after skipping
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 seed=0, drop_last=False, start_index=0):
        """
        Args:
            start_index: Number of samples to skip (per rank). This should be
                        batch_idx * batch_size from the checkpoint.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index = start_index
        self._original_num_samples = self.num_samples
    
    def set_start_index(self, start_index: int):
        """Set the number of samples to skip (for resuming mid-epoch)."""
        self.start_index = start_index
    
    def __iter__(self):
        # Get the full list of indices from parent class
        indices = list(super().__iter__())
        
        # Skip the first start_index samples
        if self.start_index > 0 and self.start_index < len(indices):
            indices = indices[self.start_index:]
        
        return iter(indices)
    
    def __len__(self):
        # Return the number of samples we'll actually yield
        remaining = self._original_num_samples - self.start_index
        return max(0, remaining)


from kermt.data.kermtdataset import (
    get_data, split_data, 
    KermtCollator, KermtDecoderCollator, KermtHybridCollator,
    get_pretokenized_data, KermtPreTokenizedDecoderCollator,
    # Memory-mapped feature classes for multi-worker hybrid/vocab training
    get_pretokenized_data_with_features,  # For hybrid mode (needs tokens + features)
    get_features_only_data,  # For vocab mode (only needs features, more efficient)
    KermtVocabFeaturesOnlyCollator,  # Optimized collator for vocab mode 
    KermtHybridPreTokenizedCollator,
    KermtVocabPreTokenizedCollator
)
from kermt.util.utils import create_logger
from kermt.util.ddp_utils import configure_nccl_for_topology, ddp_setup
from kermt.model.models import KERMTEmbedding
from task.kermttrainer import KERMTTrainer, KERMTCMIMTrainer, KERMTHybridTrainer
from kermt.util.scheduler import NoamLR
from kermt.data.torchvocab import MolVocab, SMILESVocab
from kermt.data.kermtdataset import BatchMolDataset
from kermt.util.parsing import parse_args_ddp
from kermt.util.nn_utils import param_count_trainable, param_count_total

def pre_load_data_ddp(dataset: BatchMolDataset, dataset_size: int, samples_per_file: int):
    for i in range(1, dataset_size, samples_per_file):
        dataset.load_data(i)

def setup_seed(seed: int):
    """
    Seed every RNG pretraining draws from, so a run is reproducible.

    Mirrors ``setup()`` in main.py, which finetuning has always called. Seeding the
    RNGs alone is not enough: it leaves cuBLAS/cuDNN free to pick nondeterministic
    kernels, which on a 2-epoch fixture still moved the validation loss in the
    fourth decimal. ``use_deterministic_algorithms`` is what closes that gap.

    It is safe to enable here precisely because seeding is opt-in -- a run that
    passes no ``--seed`` never reaches this function, so the path pretraining has
    always taken is untouched. ``CUBLAS_WORKSPACE_CONFIG`` must be set before the
    CUDA context is created, which is why it is handled in ``__main__`` rather than
    here.

    Every rank is seeded identically. DDP requires identical initial weights
    anyway, and the ranks see different data because the sampler shards it.

    :param seed: the seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def seed_dataloader_worker(worker_id: int):
    """
    Seed numpy and python RNGs inside a DataLoader worker process.

    torch reseeds its own RNG per worker, but not numpy's or python's. The vocab
    masking (``atom_random_mask`` / ``bond_random_mask``) and the bond dropout in
    ``MolGraph`` all draw from ``np.random``, so with ``--num_dataloader_workers > 0``
    those draws would stay unseeded and the run would not reproduce even though
    the main process was seeded.

    :param worker_id: supplied by the DataLoader; unused, since torch has already
    folded it into ``torch.initial_seed()``.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    
    # Set up signal handlers for graceful shutdown on cluster time limits
    if rank == 0:
        setup_signal_handlers()

    # parse args
    args = parse_args_ddp()

    # Opt-in: unset (the default) leaves every RNG untouched, which is how
    # pretraining has always run.
    if args.seed is not None:
        setup_seed(args.seed)

    if rank == 0:
        print(f"{args=}")
    logger = create_logger(name='pretrain', save_dir=args.save_dir)

    # Build train, val, and test datasets
    # For CMIM-only pretraining, skip loading feature files (not used by decoder)
    # For hybrid and vocab modes, we need features for functional group prediction
    load_features = args.pretrain_mode != 'cmim'
    
    # Check if using memory-mapped data (memory-efficient training)
    # CMIM mode: only needs tokens_dir (pre-tokenized SMILES)
    # Hybrid mode: needs both tokens_dir AND features_mmap_dir
    # Vocab mode: only needs features_mmap_dir (tokens not used!)
    has_tokens_dir = hasattr(args, 'tokens_dir') and args.tokens_dir is not None
    has_features_mmap = hasattr(args, 'features_mmap_dir') and args.features_mmap_dir is not None
    
    use_pretokenized_cmim = (args.pretrain_mode == 'cmim' and has_tokens_dir)
    use_features_only_vocab = (args.pretrain_mode == 'vocab' and has_features_mmap)  # Optimized: no tokens needed
    use_pretokenized_with_features = (args.pretrain_mode == 'hybrid' and has_tokens_dir and has_features_mmap)
    
    # Track whether val uses memory-mapped data (needed for collator selection)
    val_use_pretokenized = False
    val_use_features_only = False
    
    if use_features_only_vocab:
        # Optimized vocab mode: only load features and SMILES (no pre-tokenized SMILES)
        # This is more efficient than use_pretokenized_with_features for vocab mode
        if rank == 0:
            print(f"[INFO] Using features-only data for vocab training (optimized mode)")
            print(f"[INFO] Features mmap directory: {args.features_mmap_dir}")
            print(f"[INFO] SMILES cache size: {args.max_cached_files} files")

        train_features_mmap_dir = args.features_mmap_dir
        train_data, train_sample_per_file = get_features_only_data(
            data_path=args.train_data_path,
            features_mmap_dir=train_features_mmap_dir,
            max_smiles_cache_files=args.max_cached_files
        )
        train_data_size = len(train_data)
        if rank == 0:
            print(f"Training data size: {train_data_size}")
        
        if args.val_data_path is not None:
            # Infer val directory: replace 'train' with 'val' in path
            val_features_mmap_dir = args.features_mmap_dir.replace('/train/', '/val/')
            if val_features_mmap_dir == args.features_mmap_dir:
                val_features_mmap_dir = os.path.join(os.path.dirname(args.features_mmap_dir), 'val', 'feature_mmap')
            
            if os.path.exists(val_features_mmap_dir):
                val_data, val_sample_per_file = get_features_only_data(
                    data_path=args.val_data_path,
                    features_mmap_dir=val_features_mmap_dir,
                    max_smiles_cache_files=args.max_cached_files
                )
                val_data_size = len(val_data)
                val_use_features_only = True
                if rank == 0:
                    print(f"Validation data size: {val_data_size}")
            else:
                if rank == 0:
                    print(f"[WARNING] Val features directory not found: {val_features_mmap_dir}")
                    print("[WARNING] Falling back to standard data loading for validation")
                val_data, val_sample_per_file = get_data(data_path=args.val_data_path, load_features=load_features)
                val_data_size = len(val_data)
                val_use_features_only = False
                pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
        else:
            val_data = None
            val_data_size = 0
    
    elif use_pretokenized_with_features:
        # Memory-efficient loading with pre-tokenized .npy files AND memory-mapped features
        # This enables multi-worker data loading for hybrid training
        if rank == 0:
            print(f"[INFO] Using pre-tokenized data with memory-mapped features for {args.pretrain_mode} training")
            print(f"[INFO] Tokens directory: {args.tokens_dir}")
            print(f"[INFO] Features mmap directory: {args.features_mmap_dir}")
            print(f"[INFO] SMILES cache size: {args.max_cached_files} files")

        train_tokens_dir = args.tokens_dir
        train_features_mmap_dir = args.features_mmap_dir
        train_data, train_sample_per_file = get_pretokenized_data_with_features(
            data_path=args.train_data_path,
            tokens_dir=train_tokens_dir,
            features_mmap_dir=train_features_mmap_dir,
            max_smiles_cache_files=args.max_cached_files
        )
        train_data_size = len(train_data)
        if rank == 0:
            print(f"Training data size: {train_data_size}")
        
        if args.val_data_path is not None:
            # Infer val directories: replace 'train' with 'val' in paths
            val_tokens_dir = args.tokens_dir.replace('/train/', '/val/')
            val_features_mmap_dir = args.features_mmap_dir.replace('/train/', '/val/')
            
            if val_tokens_dir == args.tokens_dir:
                val_tokens_dir = os.path.join(os.path.dirname(args.tokens_dir), 'val', 'tokens')
            if val_features_mmap_dir == args.features_mmap_dir:
                val_features_mmap_dir = os.path.join(os.path.dirname(args.features_mmap_dir), 'val', 'feature_mmap')
            
            if os.path.exists(val_tokens_dir) and os.path.exists(val_features_mmap_dir):
                val_data, val_sample_per_file = get_pretokenized_data_with_features(
                    data_path=args.val_data_path,
                    tokens_dir=val_tokens_dir,
                    features_mmap_dir=val_features_mmap_dir,
                    max_smiles_cache_files=args.max_cached_files
                )
                val_data_size = len(val_data)
                val_use_pretokenized = True
                if rank == 0:
                    print(f"Validation data size: {val_data_size}")
            else:
                if rank == 0:
                    print(f"[WARNING] Val directories not found (tokens: {val_tokens_dir}, features: {val_features_mmap_dir})")
                    print("[WARNING] Falling back to standard data loading for validation")
                val_data, val_sample_per_file = get_data(data_path=args.val_data_path, load_features=load_features)
                val_data_size = len(val_data)
                val_use_pretokenized = False
                pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
        else:
            val_data = None
            val_data_size = 0
    
    elif use_pretokenized_cmim:
        # Memory-efficient loading with pre-tokenized .npy files
        if rank == 0:
            print("[INFO] Using pre-tokenized data for CMIM training (memory-efficient mode)")
            print(f"[INFO] Tokens directory: {args.tokens_dir}")
            print(f"[INFO] SMILES cache size: {args.max_cached_files} files")

        # Construct tokens_dir for train and val
        train_tokens_dir = args.tokens_dir
        train_data, train_sample_per_file = get_pretokenized_data(
            data_path=args.train_data_path,
            tokens_dir=train_tokens_dir,
            max_smiles_cache_files=args.max_cached_files
        )
        train_data_size = len(train_data)
        if rank == 0:
            print(f"Training data size: {train_data_size}")
        # No pre-loading needed - memory-mapped files are loaded on-demand
        
        if args.val_data_path is not None:
            # Infer val tokens dir: replace 'train' with 'val' in tokens_dir
            val_tokens_dir = args.tokens_dir.replace('/train/', '/val/')
            if val_tokens_dir == args.tokens_dir:
                # Fall back: assume tokens_dir is just the split directory
                val_tokens_dir = os.path.join(os.path.dirname(args.tokens_dir), 'val', 'tokens')
            
            if os.path.exists(val_tokens_dir):
                val_data, val_sample_per_file = get_pretokenized_data(
                    data_path=args.val_data_path,
                    tokens_dir=val_tokens_dir,
                    max_smiles_cache_files=args.max_cached_files
                )
                val_data_size = len(val_data)
                val_use_pretokenized = True
                if rank == 0:
                    print(f"Validation data size: {val_data_size}")
            else:
                if rank == 0:
                    print(f"[WARNING] Val tokens dir not found: {val_tokens_dir}")
                    print("[WARNING] Falling back to standard data loading for validation")
                val_data, val_sample_per_file = get_data(data_path=args.val_data_path, load_features=False)
                val_data_size = len(val_data)
                val_use_pretokenized = False
                pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
        else:
            val_data = None
            val_data_size = 0
    else:
        # Standard data loading - with optional lazy loading for large datasets
        max_cached = args.max_cached_files if args.lazy_loading else 0  # 0 = no limit when pre-loading
        
        train_data, train_sample_per_file = get_data(
            data_path=args.train_data_path, 
            load_features=load_features,
            max_cached_files=max_cached
        )
        train_data_size = len(train_data)
        print(f"Training data size: {train_data_size}")
        
        # Pre-load data unless lazy_loading is enabled (for very large datasets)
        if args.lazy_loading:
            if rank == 0:
                print(f"[INFO] Lazy loading enabled - skipping data pre-load. LRU cache size: {args.max_cached_files} files")
                if args.num_dataloader_workers > 0:
                    print(f"[WARNING] lazy_loading with num_dataloader_workers={args.num_dataloader_workers} > 0 "
                          f"may cause duplicate caching across workers. "
                          f"Consider using --num_dataloader_workers 0 or 1 for memory efficiency.")
        else:
            pre_load_data_ddp(train_data, train_data_size, train_sample_per_file)

        if args.val_data_path is not None:
            val_data, val_sample_per_file = get_data(
                data_path=args.val_data_path, 
                load_features=load_features,
                max_cached_files=max_cached
            )
            val_data_size = len(val_data)
            print(f"Validation data size: {val_data_size}")
            if not args.lazy_loading:
                pre_load_data_ddp(val_data, val_data_size, val_sample_per_file)
        else:
            val_data = None
            val_data_size = 0

    if args.test_data_path is not None:
        raise NotImplementedError("Test data is not implemented")
        test_data, test_sample_per_file = get_data(data_path=args.test_data_path, load_features=load_features)
        test_data_size = len(test_data)
        print(f"Test data size: {test_data_size}")
        pre_load_data_ddp(test_data, test_data_size, test_sample_per_file)
    else:
        test_data = None
        test_data_size = 0

    train_sampler = ResumableDistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True)
    
    if args.val_data_path is not None:
        val_sampler = DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None

    if args.test_data_path is not None:
        test_sampler = DistributedSampler(
            test_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        test_sampler = None

    # Build collator based on training mode
    shared_dict = {}
    
    if args.pretrain_mode == 'hybrid':
        # Hybrid mode - needs both SMILES vocab and atom/bond vocabularies
        if args.smiles_vocab_path is None:
            raise ValueError(
                "Hybrid training (--pretrain_mode hybrid) requires --smiles_vocab_path\n"
                "Example: --smiles_vocab_path path/to/pretrain_smiles_vocab.pkl"
            )
        if args.atom_vocab_path is None or args.bond_vocab_path is None:
            raise ValueError(
                "Hybrid training (--pretrain_mode hybrid) requires --atom_vocab_path and --bond_vocab_path\n"
                "Example: --atom_vocab_path path/to/pretrain_atom_vocab.json "
                "--bond_vocab_path path/to/pretrain_bond_vocab.json"
            )
        if rank == 0:
            print("[INFO] Hybrid mode: Loading all vocabularies")
        smiles_vocab = SMILESVocab.load_vocab(args.smiles_vocab_path)
        smiles_vocab_size = len(smiles_vocab)
        atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
        bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
        atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)
        if rank == 0:
            print(f"[INFO] SMILES vocabulary size: {smiles_vocab_size}")
            print(f"[INFO] Atom vocabulary size: {atom_vocab_size}, Bond vocabulary size: {bond_vocab_size}")
        
        # Hybrid training - use appropriate collator based on data type
        if use_pretokenized_with_features:
            if rank == 0:
                print("[INFO] Using KermtHybridPreTokenizedCollator for train (memory-efficient multi-worker)")
            mol_collator = KermtHybridPreTokenizedCollator(
                shared_dict=shared_dict,
                smiles_vocab=smiles_vocab,
                atom_vocab=atom_vocab,
                bond_vocab=bond_vocab,
                args=args
            )
            if val_use_pretokenized:
                val_collator = mol_collator
            else:
                if rank == 0 and val_data is not None:
                    print("[INFO] Using KermtHybridCollator for val (standard data)")
                val_collator = KermtHybridCollator(
                    shared_dict=shared_dict,
                    smiles_vocab=smiles_vocab,
                    atom_vocab=atom_vocab,
                    bond_vocab=bond_vocab,
                    args=args
                )
        else:
            mol_collator = KermtHybridCollator(
                shared_dict=shared_dict,
                smiles_vocab=smiles_vocab,
                atom_vocab=atom_vocab,
                bond_vocab=bond_vocab,
                args=args
            )
            val_collator = mol_collator
    elif args.pretrain_mode == 'cmim':
        # CMIM mode - only needs SMILES vocabulary
        if args.smiles_vocab_path is None:
            raise ValueError(
                "CMIM training (--pretrain_mode cmim) requires --smiles_vocab_path\n"
                "Example: --smiles_vocab_path path/to/pretrain_smiles_vocab.pkl"
            )
        if rank == 0:
            print("[INFO] CMIM mode: Loading SMILES vocabulary")
        smiles_vocab = SMILESVocab.load_vocab(args.smiles_vocab_path)
        smiles_vocab_size = len(smiles_vocab)
        if rank == 0:
            print(f"[INFO] SMILES vocabulary size: {smiles_vocab_size}")
        
        # CMIM training - use appropriate collator based on data type
        if use_pretokenized_cmim:
            if rank == 0:
                print("[INFO] Using KermtPreTokenizedDecoderCollator for train (memory-efficient)")
            mol_collator = KermtPreTokenizedDecoderCollator(
                shared_dict=shared_dict,
                smiles_vocab=smiles_vocab,
                args=args
            )
            # Val may use different collator if it fell back to standard loading
            if val_use_pretokenized:
                val_collator = mol_collator
            else:
                if rank == 0 and val_data is not None:
                    print("[INFO] Using KermtDecoderCollator for val (standard data)")
                val_collator = KermtDecoderCollator(
                    shared_dict=shared_dict,
                    smiles_vocab=smiles_vocab,
                    args=args
                )
        else:
            mol_collator = KermtDecoderCollator(
                shared_dict=shared_dict,
                smiles_vocab=smiles_vocab,
                args=args
            )
            val_collator = mol_collator
    else:  # args.pretrain_mode == 'vocab'
        # Vocab-based pretraining mode - only needs atom and bond vocabularies
        if args.atom_vocab_path is None or args.bond_vocab_path is None:
            raise ValueError(
                "Vocab-based pretraining (--pretrain_mode vocab) requires --atom_vocab_path and --bond_vocab_path\n"
                "Example: --atom_vocab_path path/to/pretrain_atom_vocab.json "
                "--bond_vocab_path path/to/pretrain_bond_vocab.json"
            )
        if rank == 0:
            print("[INFO] Vocab-based mode: Loading atom and bond vocabularies")
        atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
        bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
        atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)
        if rank == 0:
            print(f"[INFO] Atom vocabulary size: {atom_vocab_size}, Bond vocabulary size: {bond_vocab_size}")
        
        # Vocab-based pretraining - use appropriate collator based on data type
        if use_features_only_vocab:
            # Optimized: uses features-only data (no pre-tokenized SMILES loaded)
            if rank == 0:
                print("[INFO] Using KermtVocabFeaturesOnlyCollator for train (optimized multi-worker)")
            mol_collator = KermtVocabFeaturesOnlyCollator(
                shared_dict=shared_dict,
                atom_vocab=atom_vocab,
                bond_vocab=bond_vocab,
                args=args
            )
            if val_use_features_only:
                val_collator = mol_collator
            else:
                if rank == 0 and val_data is not None:
                    print("[INFO] Using KermtCollator for val (standard data)")
                val_collator = KermtCollator(
                    shared_dict=shared_dict,
                    atom_vocab=atom_vocab,
                    bond_vocab=bond_vocab,
                    args=args
                )
        else:
            # Standard mode: uses BatchMolDataset
            mol_collator = KermtCollator(
                shared_dict=shared_dict,
                atom_vocab=atom_vocab,
                bond_vocab=bond_vocab,
                args=args
            )
            val_collator = mol_collator

    # Only set when seeding is requested, so the unseeded default path is untouched.
    worker_init_fn = seed_dataloader_worker if args.seed is not None else None

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, # batch size per GPU (aka micro batch size)
                                  shuffle=False, # because train_sampler does the shuffling
                                  num_workers=args.num_dataloader_workers,
                                  sampler=train_sampler,
                                  collate_fn=mol_collator,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=True)
    
    if args.val_data_path is not None:
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, 
                                  shuffle=False, # because no shuffling needed
                                  num_workers=args.num_dataloader_workers,
                                  sampler=val_sampler,
                                  collate_fn=val_collator,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=True)
    else:
        val_dataloader = None
        
    # Build model - create complete task model based on training mode
    # This ensures all model parameters (encoder, decoder, heads) are included
    kermt_embedding = KERMTEmbedding(args)
    fg_size = 85  # Fixed size for functional group labels
    
    if args.pretrain_mode == 'hybrid':
        # Build hybrid task model (encoder + latent + decoder + vocab heads)
        if rank == 0:
            print("[INFO] Building KermtHybridTask model")
        from kermt.model.models import KermtHybridTask
        model = KermtHybridTask(
            args,
            kermt=kermt_embedding,
            latent_dim=args.latent_dim,
            contrastive_temperature=args.contrastive_temperature,
            smiles_vocab_size=smiles_vocab_size,
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            fg_size=fg_size
        )
    elif args.pretrain_mode == 'cmim':
        # Build complete CMIM task model (encoder + latent distribution + decoder)
        if rank == 0:
            print("[INFO] Building KermtCMIMTask model")
        from kermt.model.models import KermtCMIMTask
        model = KermtCMIMTask(
            args,
            kermt=kermt_embedding,
            latent_dim=args.latent_dim,
            contrastive_temperature=args.contrastive_temperature,
            smiles_vocab_size=smiles_vocab_size
        )
    else:  # args.pretrain_mode == 'vocab'
        # Build complete vocab task model (encoder + vocab prediction heads)
        if rank == 0:
            print("[INFO] Building KermtTask model")
        from kermt.model.models import KermtTask
        model = KermtTask(
            args,
            kermt=kermt_embedding,
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
            fg_size=fg_size
        )
    
    print(f'Number of trainable parameters = {param_count_trainable(model):,}')
    print(f'Number of total parameters = {param_count_total(model):,}')

    # Build optimizer on COMPLETE model (includes all trainable components)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    # Build Learning rate scheduler   
    steps_per_epoch = train_data_size // (args.batch_size*world_size)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        fine_tune_coff=args.fine_tune_coff
    )
 
    # Build trainer - pass the complete model
    if args.pretrain_mode == 'hybrid':
        if rank == 0:
            print("[INFO] Initializing KERMTHybridTrainer")
        trainer = KERMTHybridTrainer(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu_id=rank,
            n_steps=0,
            logger=logger,
            shutdown_checker=is_shutdown_requested
        )
    elif args.pretrain_mode == 'cmim':
        if rank == 0:
            print("[INFO] Initializing KERMTCMIMTrainer")
        trainer = KERMTCMIMTrainer(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu_id=rank,
            n_steps=0,
            logger=logger,
            shutdown_checker=is_shutdown_requested
        )
    else:  # args.pretrain_mode == 'vocab'
        if rank == 0:
            print("[INFO] Initializing KERMTTrainer (vocab-based pretraining)")
        trainer = KERMTTrainer(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gpu_id=rank,
            n_steps=0,
            logger=logger,
            shutdown_checker=is_shutdown_requested
        )

    wandb_run_id = None
    if args.save_dir is not None:
        last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pt")
        if os.path.exists(last_ckpt_path):
            print(f"Loading checkpoint from {last_ckpt_path}")
            epoch, scheduler_step, prev_batch_idx, wandb_run_id = trainer.load(last_ckpt_path)
            print(f"Loaded checkpoint from epoch={epoch}, scheduler_step={scheduler_step}, prev_batch_idx={prev_batch_idx}")
        else:
            epoch = 0
            scheduler_step = 0
            prev_batch_idx = 0

    steps_per_epoch = train_data_size // (args.batch_size*world_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Resume mid-epoch by setting sampler start index (efficient - no data loading for skipped samples)
    # This works because:
    # 1. set_epoch() ensures deterministic shuffle order (called in trainer.iter())
    # 2. The sampler skips indices at the index level, not by loading and discarding
    # 3. The trainer doesn't need to skip batches - sampler handles it
    if prev_batch_idx > 0:
        # Calculate samples to skip: batch_idx * batch_size
        samples_to_skip = prev_batch_idx * args.batch_size
        train_sampler.set_start_index(samples_to_skip)
        if rank == 0:
            print(f"[INFO] Resuming mid-epoch: skipping first {prev_batch_idx} batches ({samples_to_skip} samples)")
            print(f"[INFO] Sampler will yield {len(train_sampler)} remaining batches")
    
    # Tell trainer:
    # - batch_idx=0: don't skip batches in training loop (sampler handles it)
    # - batch_idx_offset=prev_batch_idx: add this when saving so checkpoint has correct batch_idx
    trainer.set_batch_idx(0, batch_idx_offset=prev_batch_idx)

    # Initialize WandB on rank 0 (before training starts so resumed runs attach correctly)
    if rank == 0 and getattr(args, 'wandb_project', None):
        try:
            import wandb
            wandb.login()
            if wandb_run_id:
                print(f"[INFO] Resuming WandB run: {wandb_run_id}", flush=True)
            wandb.init(
                project=args.wandb_project,
                name=getattr(args, 'wandb_run_name', None),
                id=wandb_run_id,  # None for new runs, restored from checkpoint for restarts
                config=vars(args),
                dir=args.save_dir if hasattr(args, 'save_dir') else None,
                resume="allow"
            )
            print(f"[INFO] WandB initialized: project={args.wandb_project}, run_id={wandb.run.id}", flush=True)
        except ImportError:
            print("[WARNING] wandb not installed, skipping WandB logging", flush=True)
            args.wandb_project = None
        except Exception as e:
            print(f"[WARNING] WandB initialization failed: {e}", flush=True)
            print("[WARNING] Continuing without WandB logging", flush=True)
            args.wandb_project = None

    # Train model
    trainer.train(start_epoch=epoch, max_epochs=args.epochs)

    # Finish WandB run
    if rank == 0 and getattr(args, 'wandb_project', None):
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    destroy_process_group()


if __name__ == "__main__":

    world_size = os.environ.get("WORLD_SIZE", 1)
    world_size = int(world_size)
    print(f"World size: {world_size}")

    # setup_seed() enables deterministic algorithms, and cuBLAS requires this to be
    # set before the CUDA context exists -- so it has to happen here, in the parent,
    # before mp.spawn. Set unconditionally: it is inert unless deterministic
    # algorithms are actually switched on, and reading --seed this early would mean
    # parsing args before the DDP ranks do. An existing value is left alone.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    # Auto-configure NCCL before spawning processes
    # This detects GPU topology and sets appropriate P2P settings
    configure_nccl_for_topology()
    
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
