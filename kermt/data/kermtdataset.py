"""
The dataset used in training KERMT.

File Organization:
==================
1. Imports
2. Shared Collator Helper Functions
3. Standard Data Loading (BatchDatapoint, BatchMolDataset, standard collators)
4. Memory-Mapped CMIM (PreTokenizedBatch*, KermtPreTokenizedDecoderCollator)
5. Memory-Mapped Hybrid (PreTokenizedWithFeatures*, KermtHybridPreTokenizedCollator)
6. Memory-Mapped Vocab-Only (FeaturesOnly*, KermtVocabFeaturesOnlyCollator)
"""
import math
import os
import csv
from typing import Union, List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem

import kermt.util.utils as feautils
from kermt.data import mol2graph
from kermt.data.moldataset import MoleculeDatapoint
from kermt.data.task_labels import atom_to_vocab, bond_to_vocab
from kermt.util.features import get_feature_range

import cuik_molmaker
from kermt.util._cuik_compat import (
    atom_onehot_feature_names_to_tensor,
    atom_float_feature_names_to_tensor,
    bond_feature_names_to_tensor,
)


# ============================================================================
# SECTION 1: Shared Collator Helper Functions
# ============================================================================

def setup_cuik_molmaker_features(args):
    """
    Helper function to set up cuik-molmaker feature tensors and ranges.
    Shared by all collators.
    
    Args:
        args: Training arguments with use_cuikmolmaker_featurization flag
        
    Returns:
        tuple: (cmm_feature_tensors dict, cmm_feature_range) or (None, None)
    """
    if not args.use_cuikmolmaker_featurization:
        return None, None
    
    # Define feature properties
    atom_onehot_props = [
        "atomic-number", "total-degree", "formal-charge", "chirality",
        "num-hydrogens", "hybridization", "implicit-valence", "ring-size"
    ]
    atom_float_props = [
        "aromatic", "mass", "hydrogen-bond-acceptor",
        "hydrogen-bond-donor", "acidic", "basic"
    ]
    bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
    
    # Form feature arrays via the version-agnostic compat shim
    # (kermt:latest container ships the post-0.2 `_to_tensor` rename;
    # admet_dev_py311 host conda env still ships 0.2 with `_to_array`).
    cmm_feature_tensors = {
        "atom_onehot": atom_onehot_feature_names_to_tensor(atom_onehot_props),
        "atom_float": atom_float_feature_names_to_tensor(atom_float_props),
        "bond": bond_feature_names_to_tensor(bond_props),
    }
    
    # Get feature ranges
    cmm_feature_range = get_feature_range(atom_onehot_props, atom_float_props)
    
    return cmm_feature_tensors, cmm_feature_range


def atom_random_mask(smiles_batch, atom_vocab, percent=0.15):
    """
    Perform random mask operation on atoms for vocabulary prediction.
    Shared by vocab-based collators.
    
    Args:
        smiles_batch: List of SMILES strings
        atom_vocab: MolVocab instance for atom vocabulary
        percent: Fraction of atoms to mask (default 0.15)
        
    Returns:
        List of atom vocabulary labels (0 = not masked)
    """
    vocab_label = [0]  # Zero padding at start
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        mlabel = [0] * mol.GetNumAtoms()
        n_mask = math.ceil(mol.GetNumAtoms() * percent)
        perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
        for p in perm:
            atom = mol.GetAtomWithIdx(int(p))
            mlabel[p] = atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index)
        vocab_label.extend(mlabel)
    return vocab_label


def bond_random_mask(smiles_batch, bond_vocab, percent=0.15, rdkit_bond_idx=None):
    """
    Perform random mask operation on bonds for vocabulary prediction.
    Shared by vocab-based collators.

    The bond vocab head is supervised per bond, and the labels are matched to the graph
    by position alone. This function re-parses the SMILES independently of ``MolGraph``,
    so when ``--bond_drop_rate > 0`` drops bonds from the graph the two disagree: the
    label vector is too long and every label after the first dropped bond describes a
    different bond from the one it lands on.

    ``rdkit_bond_idx`` closes that gap. Labels are collected into a table keyed by RDKit
    bond index, then gathered in the order the surviving bonds actually appear in the
    packed batch, so dropped bonds simply do not contribute a label.

    The masking loop below is unchanged, including the order it visits bonds in and the
    single ``np.random.permutation`` draw per molecule, so a run with no bond dropout
    produces exactly the labels it produced before -- the gather is the identity in that
    case.

    Args:
        smiles_batch: List of SMILES strings
        bond_vocab: MolVocab instance for bond vocabulary
        percent: Fraction of bonds to mask (default 0.15)
        rdkit_bond_idx: ``BatchMolGraph.rdkit_bond_idx`` -- packed directed-bond row to
            global RDKit bond index, two entries per surviving bond. When None the labels
            are returned in the legacy sequential order, which is only correct if no bond
            was dropped.

    Returns:
        List of bond vocabulary labels (0 = not masked), with a leading padding element.
        One label per surviving undirected bond, matching the rows
        :class:`BondVocabPrediction` emits.
    """
    label_by_rdkit_idx = []   # global RDKit bond index -> label
    sequential_label = []     # legacy order: one label per bond as visited

    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        nm_atoms = mol.GetNumAtoms()
        nm_bonds = mol.GetNumBonds()
        mlabel = [0] * nm_bonds
        n_mask = math.ceil(nm_bonds * percent)
        perm = np.random.permutation(nm_bonds)[:n_mask]
        virtual_bond_id = 0
        for a1 in range(nm_atoms):
            for a2 in range(a1 + 1, nm_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                if virtual_bond_id in perm:
                    label = bond_vocab.stoi.get(bond_to_vocab(mol, bond), bond_vocab.other_index)
                else:
                    label = 0
                # Keyed by RDKit index so it can be gathered against the packed batch;
                # also appended in visit order for the legacy return.
                mlabel[bond.GetIdx()] = label
                sequential_label.append(label)
                virtual_bond_id += 1
        # extend keeps the per-molecule offsets identical to BatchMolGraph's,
        # which accumulates mol.GetNumBonds() the same way.
        label_by_rdkit_idx.extend(mlabel)

    if rdkit_bond_idx is None:
        return [0] + sequential_label

    # Two entries per surviving bond (one per direction); take one per bond, in packed
    # order, to match BondVocabPrediction's output rows.
    return [0] + [label_by_rdkit_idx[gidx] for gidx in rdkit_bond_idx[0::2]]


def tokenize_and_pad_smiles(smiles_batch, smiles_vocab, max_seq_len):
    """
    Tokenize SMILES strings, truncate if needed, and pad to same length.
    Shared by decoder-based collators.
    
    Args:
        smiles_batch: List of SMILES strings
        smiles_vocab: SMILESVocab instance for tokenization
        max_seq_len: Maximum sequence length
        
    Returns:
        tokens: [batch_size, max_len] padded token IDs with <start> and <end>
        lengths: [batch_size] sequence lengths after truncation
        padding_mask: [batch_size, max_len] True where padded
    """
    batch_size = len(smiles_batch)
    
    # Tokenize all SMILES (includes <start> and <end>)
    tokenized = [
        smiles_vocab.smiles_to_ids(smi, add_special_tokens=True)
        for smi in smiles_batch
    ]
    
    # Truncate sequences that exceed max_seq_len
    end_token_id = smiles_vocab.end_index
    for i, ids in enumerate(tokenized):
        if len(ids) > max_seq_len:
            tokenized[i] = ids[:max_seq_len-1] + [end_token_id]
    
    # Get lengths (after truncation)
    lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)
    
    # Pad to same length
    max_len = max(lengths)
    pad_idx = smiles_vocab.pad_index
    
    tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    for i, ids in enumerate(tokenized):
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    
    # Create padding mask (True where padding)
    padding_mask = tokens == pad_idx
    
    return tokens, lengths, padding_mask


def prepare_decoder_sequences(tokens):
    """
    Prepare decoder input and target sequences for teacher forcing.
    Shared by decoder-based collators.
    
    Input:  [<start>, C, C, (, =, O, ), <end>, <pad>, <pad>]
    Decoder input:  [<start>, C, C, (, =, O, ), <end>, <pad>]  (all except last)
    Decoder target: [C, C, (, =, O, ), <end>, <pad>, <pad>]     (all except first)
    
    Args:
        tokens: [batch, seq_len] full sequences with <start> and <end>
        
    Returns:
        decoder_input: [batch, seq_len-1] input to decoder
        decoder_target: [batch, seq_len-1] target for loss
    """
    decoder_input = tokens[:, :-1].contiguous()
    decoder_target = tokens[:, 1:].contiguous()
    return decoder_input, decoder_target


def create_causal_mask(seq_len, positional_encoding='rope', device='cpu'):
    """
    Create causal mask for autoregressive generation.
    Shared by decoder-based collators.
    
    Args:
        seq_len: Sequence length
        positional_encoding: Type of positional encoding ('rope' or 'sinusoidal')
        device: Device to create mask on
        
    Returns:
        mask: [seq_len, seq_len] causal mask in appropriate format
    """
    if positional_encoding == 'sinusoidal':
        # PyTorch's nn.TransformerDecoder expects BoolTensor
        # True = masked (cannot attend), False = allowed
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    elif positional_encoding == 'rope':
        # Custom RoPE implementation expects FloatTensor with additive masking
        # -inf = masked (becomes 0 after softmax), 0 = allowed
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
    else:
        raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
    
    return mask


def _build_graph_input(smiles_batch, shared_dict, args, cmm_feature_tensors=None, cmm_feature_range=None):
    """
    Build graph input from SMILES batch.
    Shared by all collators.
    
    Args:
        smiles_batch: List of SMILES strings
        shared_dict: Shared dictionary for mol2graph caching
        args: Training arguments
        cmm_feature_tensors: Cuik-molmaker feature tensors (optional)
        cmm_feature_range: Cuik-molmaker feature range (optional)
        
    Returns:
        (BatchMolGraph components, rdkit_bond_idx). The second element maps each packed
        directed-bond row to its global RDKit bond index and is what lets the vocab task
        align its labels to the bonds that survived --bond_drop_rate.
    """
    if args.use_cuikmolmaker_featurization and cmm_feature_tensors is not None:
        batch = mol2graph(smiles_batch, shared_dict, args,
                          cmm_feature_range=cmm_feature_range,
                          cmm_tensors=cmm_feature_tensors)
    else:
        batch = mol2graph(smiles_batch, shared_dict, args)
    return batch.get_components(), batch.rdkit_bond_idx


def _generate_vocab_targets(smiles_batch, atom_vocab, bond_vocab, features_list=None, batch=None,
                            rdkit_bond_idx=None):
    """
    Generate vocabulary prediction targets.
    Shared by vocab-based collators.
    
    Args:
        smiles_batch: List of SMILES strings
        atom_vocab: MolVocab for atom vocabulary
        bond_vocab: MolVocab for bond vocabulary
        features_list: Pre-computed features (for mmap mode), or None
        batch: List of MoleculeDatapoint (for standard mode), or None
        
    Returns:
        dict with av_task, bv_task, fg_task tensors
    """
    atom_vocab_label = torch.Tensor(atom_random_mask(smiles_batch, atom_vocab)).long()
    bond_vocab_label = torch.Tensor(
        bond_random_mask(smiles_batch, bond_vocab, rdkit_bond_idx=rdkit_bond_idx)).long()
    
    if features_list is not None:
        # Memory-mapped mode: features are pre-computed numpy arrays
        fgroup_label = torch.tensor(np.stack(features_list), dtype=torch.float32)
    else:
        # Standard mode: features are in MoleculeDatapoint
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
    
    return {
        "av_task": atom_vocab_label,
        "bv_task": bond_vocab_label,
        "fg_task": fgroup_label
    }


def _pad_pretokenized_sequences(tokens_list, pad_index):
    """
    Pad pre-tokenized sequences to same length.
    Shared by pre-tokenized collators.
    
    Args:
        tokens_list: List of numpy arrays with potentially different lengths
        pad_index: Padding token index
        
    Returns:
        tokens_tensor: [batch, max_len] padded tensor
        padding_mask: [batch, max_len] boolean mask (True = padding)
    """
    max_len = max(len(t) for t in tokens_list)
    batch_size = len(tokens_list)
    
    padded_tokens = np.full((batch_size, max_len), pad_index, dtype=np.int64)
    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, :len(tokens)] = tokens
    
    tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    padding_mask = torch.tensor(padded_tokens == pad_index, dtype=torch.bool)
    
    return tokens_tensor, padding_mask


# ============================================================================
# SECTION 2: Standard Data Loading
# ============================================================================
# Used when data is loaded directly from .csv and .npz files.
# Supports lazy loading with LRU cache, but limited to 1 worker due to
# pickle issues with file handles.

def get_data(data_path, logger=None, load_features=True, max_cached_files=100):
    """
    Load data from the data_path for standard training.
    
    Args:
        data_path: the data_path containing summary.txt, graph/, feature/
        logger: optional logger
        load_features: whether to load feature files (.npz). Set to False for CMIM
                      pretraining which doesn't use features.
        max_cached_files: Maximum number of files to keep in memory (LRU cache).
                         Set to 0 or None to disable cache limit.
                         
    Returns:
        (BatchMolDataset, sample_per_file)
    """
    debug = logger.debug if logger is not None else print
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    feature_path = os.path.join(data_path, "feature")

    fin = open(summary_path)
    n_files = int(fin.readline().strip().split(":")[-1])
    n_samples = int(fin.readline().strip().split(":")[-1])
    sample_per_file = int(fin.readline().strip().split(":")[-1])
    debug("Loading data:")
    debug("Number of files: %d" % n_files)
    debug("Number of samples: %d" % n_samples)
    debug("Samples/file: %d" % sample_per_file)
    debug("Load features: %s" % load_features)
    debug("Max cached files: %s" % max_cached_files)

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        feature_path_i = os.path.join(feature_path, str(i) + ".npz")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint(smiles_path_i, feature_path_i, n_samples_i, load_features))
    return BatchMolDataset(datapoints, max_cached_files=max_cached_files), sample_per_file


def split_data(data, split_type='random', sizes=(0.8, 0.1, 0.1), seed=0, logger=None):
    """
    Split data with given train/validation/test ratio.
    
    Args:
        data: BatchMolDataset to split
        split_type: 'random' (only supported type)
        sizes: (train, val, test) ratios, must sum to 1
        seed: random seed
        logger: optional logger
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == "random":
        data.shuffle(seed=seed)
        data = data.data

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return BatchMolDataset(train), BatchMolDataset(val), BatchMolDataset(test)
    else:
        raise NotImplementedError("Do not support %s splits" % split_type)


class BatchDatapoint:
    """
    A batch datapoint representing one file of SMILES and features.
    
    Used with standard (non-mmap) data loading. Supports lazy loading
    where data is only loaded when first accessed.
    """
    
    def __init__(self, smiles_file, feature_file, n_samples, load_features=True):
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        self.n_samples = n_samples
        self.load_features = load_features
        self.datapoints = None

    def load_datapoints(self):
        # Skip loading features for CMIM pretraining (features not used)
        if self.load_features:
            features = self.load_feature()
        else:
            features = [None] * self.n_samples
        self.datapoints = []

        with open(self.smiles_file) as f:
            reader = csv.reader(f)
            next(reader)
            for i, line in enumerate(reader):
                d = MoleculeDatapoint(line=line, features=features[i])
                self.datapoints.append(d)

        assert len(self.datapoints) == self.n_samples

    def load_feature(self):
        return feautils.load_features(self.feature_file)

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.datapoints is None:
            self.load_datapoints()
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchMolDataset(Dataset):
    """
    Dataset for standard data loading with LRU cache.
    
    Used with: KermtCollator, KermtDecoderCollator, KermtHybridCollator
    Returns: (MoleculeDatapoint, idx)
    """
    
    def __init__(self, data: List[BatchDatapoint], graph_per_file=None, max_cached_files=100):
        """
        Args:
            data: List of BatchDatapoint objects
            graph_per_file: Number of samples per file
            max_cached_files: Maximum number of files to keep in memory (LRU cache).
                              Set to None or 0 to disable cache limit (keep all).
        """
        self.data = data
        self.max_cached_files = max_cached_files if max_cached_files else None
        self.loaded_file_order = []  # Track order for LRU eviction

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()
        self.loaded_file_order = []

    def __len__(self) -> int:
        return self.len

    def _ensure_loaded_with_cache(self, dp_idx):
        """Load file at dp_idx, evicting old files if cache is full."""
        if self.data[dp_idx].is_loaded():
            # Already loaded - move to end of LRU list (most recently used)
            if dp_idx in self.loaded_file_order:
                self.loaded_file_order.remove(dp_idx)
            self.loaded_file_order.append(dp_idx)
            return
        
        # Need to load - first check if we need to evict
        if self.max_cached_files and len(self.loaded_file_order) >= self.max_cached_files:
            # Evict least recently used file (first in list)
            evict_idx = self.loaded_file_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        # Load the new file
        self.data[dp_idx].load_datapoints()
        self.loaded_file_order.append(dp_idx)

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        self._ensure_loaded_with_cache(dp_idx)
        return self.data[dp_idx][real_idx], idx

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        self._ensure_loaded_with_cache(dp_idx)

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res


class KermtCollator(object):
    """
    Collator for vocabulary-based pretraining (original GROVER).
    
    Used with: BatchMolDataset (standard loading)
    Input: List of (MoleculeDatapoint, idx) tuples
    Output: dict with graph_input, targets (av_task, bv_task, fg_task)
    """
    
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_idx):
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        batchgraph, rdkit_bond_idx = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                                        self.cmm_feature_tensors, self.cmm_feature_range)
        targets = _generate_vocab_targets(smiles_batch, self.atom_vocab, self.bond_vocab, batch=batch,
                                          rdkit_bond_idx=rdkit_bond_idx)
        
        return {
            "graph_input": batchgraph,
            "targets": targets,
            "idx": idx
        }


class KermtDecoderCollator(object):
    """
    Collator for CMIM training with transformer decoder.
    
    Used with: BatchMolDataset (standard loading)
    Input: List of (MoleculeDatapoint, idx) tuples
    Output: dict with graph_input, decoder_input, decoder_target, masks
    """
    
    def __init__(self, shared_dict, smiles_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_idx):
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        # Build graph input
        batchgraph, _ = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                           self.cmm_feature_tensors, self.cmm_feature_range)
        
        # Tokenize and prepare decoder sequences
        tokens, lengths, padding_mask = tokenize_and_pad_smiles(
            smiles_batch, self.smiles_vocab, self.args.decoder_max_seq_len
        )
        decoder_input, decoder_target = prepare_decoder_sequences(tokens)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        return {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "smiles": smiles_batch,
            "idx": idx
        }


class KermtHybridCollator(object):
    """
    Collator for hybrid CMIM + vocabulary pretraining.
    
    Used with: BatchMolDataset (standard loading)
    Input: List of (MoleculeDatapoint, idx) tuples
    Output: dict with graph_input, decoder inputs, vocab targets
    """
    
    def __init__(self, shared_dict, smiles_vocab, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_idx):
        batch, idx = zip(*batch_idx)
        smiles_batch = [d.smiles for d in batch]
        
        # Build graph input
        batchgraph, rdkit_bond_idx = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                                        self.cmm_feature_tensors, self.cmm_feature_range)
        
        # Vocab targets
        targets = _generate_vocab_targets(smiles_batch, self.atom_vocab, self.bond_vocab, batch=batch,
                                          rdkit_bond_idx=rdkit_bond_idx)
        
        # Decoder sequences
        tokens, lengths, padding_mask = tokenize_and_pad_smiles(
            smiles_batch, self.smiles_vocab, self.args.decoder_max_seq_len
        )
        decoder_input, decoder_target = prepare_decoder_sequences(tokens)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        return {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "targets": targets,
            "smiles": smiles_batch,
            "idx": idx
        }


# ============================================================================
# SECTION 3: Memory-Mapped Data Loading - CMIM
# ============================================================================
# Used for CMIM pretraining with large datasets.
# Memory-mapped .npy files enable multi-worker data loading.

def get_pretokenized_data(data_path, tokens_dir, logger=None, max_smiles_cache_files=50):
    """
    Load pre-tokenized data from memory-mapped .npy files for CMIM training.
    
    Args:
        data_path: Base data path (for graph/ SMILES)
        tokens_dir: Path to pre-tokenized .npy files
        logger: Optional logger
        max_smiles_cache_files: Max files to keep SMILES cached (LRU eviction)
        
    Returns:
        (PreTokenizedBatchMolDataset, sample_per_file)
    """
    debug = logger.debug if logger is not None else print
    
    summary_path = os.path.join(tokens_dir, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Make sure you've run pretokenize_zinc15.py to completion."
        )
    
    with open(summary_path) as fin:
        n_files = int(fin.readline().strip().split(":")[-1])
        n_samples = int(fin.readline().strip().split(":")[-1])
        sample_per_file = int(fin.readline().strip().split(":")[-1])
    
    debug("Loading pre-tokenized data:")
    debug(f"  Number of files: {n_files}")
    debug(f"  Number of samples: {n_samples}")
    debug(f"  Samples/file: {sample_per_file}")
    debug(f"  Tokens directory: {tokens_dir}")
    debug(f"  Max SMILES cache files: {max_smiles_cache_files}")
    
    datapoints = []
    for i in range(n_files):
        tokens_path_i = os.path.join(tokens_dir, str(i) + ".npy")
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        if n_samples_i == 0:
            n_samples_i = sample_per_file
        datapoints.append(PreTokenizedBatchDatapoint(tokens_path_i, smiles_path_i, n_samples_i))
    
    return PreTokenizedBatchMolDataset(datapoints, max_smiles_cache_files=max_smiles_cache_files), sample_per_file


class PreTokenizedBatchDatapoint:
    """
    A batch datapoint with memory-mapped pre-tokenized tokens.
    
    Uses np.load(mmap_mode='r') so the same physical memory is shared
    across all DataLoader workers via OS page cache.
    """
    
    def __init__(self, tokens_file: str, smiles_file: str, n_samples: int):
        self.tokens_file = tokens_file
        self.smiles_file = smiles_file
        self.n_samples = n_samples
        self._tokens_mmap = None
        self._smiles_cache = None
    
    def _ensure_mmap(self):
        if self._tokens_mmap is None:
            self._tokens_mmap = np.load(self.tokens_file, mmap_mode='r')
    
    def _ensure_smiles(self):
        if self._smiles_cache is None:
            self._smiles_cache = []
            with open(self.smiles_file) as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    self._smiles_cache.append(line[0])
    
    def get_tokens(self, idx: int) -> np.ndarray:
        self._ensure_mmap()
        return self._tokens_mmap[idx]
    
    def get_smiles(self, idx: int) -> str:
        self._ensure_smiles()
        return self._smiles_cache[idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.get_tokens(idx), self.get_smiles(idx), idx
    
    def is_loaded(self):
        return True
    
    def clean_cache(self):
        self._smiles_cache = None


class PreTokenizedBatchMolDataset(Dataset):
    """
    Dataset for pre-tokenized CMIM data with memory-mapped tokens.
    
    Used with: KermtPreTokenizedDecoderCollator
    Returns: (tokens, smiles, idx)
    """
    
    def __init__(self, data: List[PreTokenizedBatchDatapoint], graph_per_file=None,
                 max_smiles_cache_files: int = 50):
        self.data = data
        self.len = sum(len(d) for d in self.data)
        self.max_smiles_cache_files = max_smiles_cache_files
        self._smiles_cache_order = []
        
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
    
    def __len__(self) -> int:
        return self.len
    
    def _ensure_smiles_with_cache(self, dp_idx: int):
        datapoint = self.data[dp_idx]
        if datapoint._smiles_cache is not None:
            if dp_idx in self._smiles_cache_order:
                self._smiles_cache_order.remove(dp_idx)
            self._smiles_cache_order.append(dp_idx)
            return
        
        if (self.max_smiles_cache_files > 0 and 
            len(self._smiles_cache_order) >= self.max_smiles_cache_files):
            evict_idx = self._smiles_cache_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        datapoint._ensure_smiles()
        self._smiles_cache_order.append(dp_idx)
    
    def __getitem__(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        self._ensure_smiles_with_cache(dp_idx)
        tokens, smiles, local_idx = self.data[dp_idx][real_idx]
        return tokens, smiles, idx
    
    def clean_cache(self):
        for d in self.data:
            d.clean_cache()
        self._smiles_cache_order = []
    
    def shuffle(self, seed: int = None):
        pass


class KermtPreTokenizedDecoderCollator(object):
    """
    Collator for CMIM training with pre-tokenized data.
    
    Used with: PreTokenizedBatchMolDataset
    Input: List of (tokens, smiles, idx) tuples
    Output: dict with graph_input, decoder inputs
    """
    
    def __init__(self, shared_dict, smiles_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_data):
        tokens_batch, smiles_batch, idx = zip(*batch_data)
        
        # Pad pre-tokenized sequences
        tokens_tensor, padding_mask = _pad_pretokenized_sequences(
            tokens_batch, self.smiles_vocab.pad_index
        )
        
        # Build graph input
        batchgraph, _ = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                           self.cmm_feature_tensors, self.cmm_feature_range)
        
        # Prepare decoder sequences
        decoder_input, decoder_target = prepare_decoder_sequences(tokens_tensor)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        return {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "smiles": smiles_batch,
            "idx": idx
        }


# ============================================================================
# SECTION 4: Memory-Mapped Data Loading - Hybrid
# ============================================================================
# Used for hybrid pretraining with large datasets.
# Requires both tokens and features in .npy format.

def get_pretokenized_data_with_features(
    data_path: str,
    tokens_dir: str,
    features_mmap_dir: str,
    logger=None,
    max_smiles_cache_files: int = 50
):
    """
    Load pre-tokenized data with memory-mapped features for hybrid training.
    
    Note: For vocab-only mode, use get_features_only_data() instead.
    
    Args:
        data_path: Base data path (for graph/ SMILES and summary.txt)
        tokens_dir: Path to pre-tokenized .npy files
        features_mmap_dir: Path to memory-mappable feature .npy files
        logger: Optional logger
        max_smiles_cache_files: Max files to keep SMILES cached
        
    Returns:
        (PreTokenizedWithFeaturesMolDataset, sample_per_file)
    """
    debug = logger.debug if logger is not None else print
    
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Make sure you've run prepare_zinc15_unified.py to completion."
        )
    
    if not os.path.exists(features_mmap_dir):
        raise FileNotFoundError(
            f"Features mmap directory not found: {features_mmap_dir}\n"
            f"Make sure you've run prepare_zinc15_unified.py which generates feature_mmap/ directory."
        )
    
    with open(summary_path) as fin:
        n_files = int(fin.readline().strip().split(":")[-1])
        n_samples = int(fin.readline().strip().split(":")[-1])
        sample_per_file = int(fin.readline().strip().split(":")[-1])
    
    debug("Loading pre-tokenized data with memory-mapped features:")
    debug(f"  Number of files: {n_files}")
    debug(f"  Number of samples: {n_samples}")
    debug(f"  Samples/file: {sample_per_file}")
    debug(f"  Tokens directory: {tokens_dir}")
    debug(f"  Features mmap directory: {features_mmap_dir}")
    debug(f"  Max SMILES cache files: {max_smiles_cache_files}")
    
    datapoints = []
    for i in range(n_files):
        tokens_path_i = os.path.join(tokens_dir, str(i) + ".npy")
        features_path_i = os.path.join(features_mmap_dir, str(i) + ".npy")
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        if n_samples_i == 0:
            n_samples_i = sample_per_file
        
        if not os.path.exists(features_path_i):
            raise FileNotFoundError(
                f"Feature file not found: {features_path_i}\n"
                f"Expected {n_files} feature files in {features_mmap_dir}"
            )
        
        datapoints.append(PreTokenizedWithFeaturesDatapoint(
            tokens_file=tokens_path_i,
            features_file=features_path_i,
            smiles_file=smiles_path_i,
            n_samples=n_samples_i
        ))
    
    return PreTokenizedWithFeaturesMolDataset(
        datapoints, max_smiles_cache_files=max_smiles_cache_files
    ), sample_per_file


class PreTokenizedWithFeaturesDatapoint:
    """
    A batch datapoint with memory-mapped tokens AND features.
    """
    
    def __init__(self, tokens_file: str, features_file: str, smiles_file: str, n_samples: int):
        self.tokens_file = tokens_file
        self.features_file = features_file
        self.smiles_file = smiles_file
        self.n_samples = n_samples
        self._tokens_mmap = None
        self._features_mmap = None
        self._smiles_cache = None
    
    def _ensure_tokens_mmap(self):
        if self._tokens_mmap is None:
            self._tokens_mmap = np.load(self.tokens_file, mmap_mode='r')
    
    def _ensure_features_mmap(self):
        if self._features_mmap is None:
            self._features_mmap = np.load(self.features_file, mmap_mode='r')
    
    def _ensure_smiles(self):
        if self._smiles_cache is None:
            self._smiles_cache = []
            with open(self.smiles_file) as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    self._smiles_cache.append(line[0])
    
    def get_tokens(self, idx: int) -> np.ndarray:
        self._ensure_tokens_mmap()
        return self._tokens_mmap[idx]
    
    def get_features(self, idx: int) -> np.ndarray:
        self._ensure_features_mmap()
        return self._features_mmap[idx]
    
    def get_smiles(self, idx: int) -> str:
        self._ensure_smiles()
        return self._smiles_cache[idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.get_tokens(idx), self.get_features(idx), self.get_smiles(idx), idx
    
    def is_loaded(self):
        return True
    
    def clean_cache(self):
        self._smiles_cache = None


class PreTokenizedWithFeaturesMolDataset(Dataset):
    """
    Dataset for pre-tokenized data with memory-mapped features.
    
    Used with: KermtHybridPreTokenizedCollator
    Returns: (tokens, features, smiles, idx)
    """
    
    def __init__(self, data: List[PreTokenizedWithFeaturesDatapoint], graph_per_file=None,
                 max_smiles_cache_files: int = 50):
        self.data = data
        self.len = sum(len(d) for d in self.data)
        self.max_smiles_cache_files = max_smiles_cache_files
        self._smiles_cache_order = []
        
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
    
    def __len__(self) -> int:
        return self.len
    
    def _ensure_smiles_with_cache(self, dp_idx: int):
        datapoint = self.data[dp_idx]
        if datapoint._smiles_cache is not None:
            if dp_idx in self._smiles_cache_order:
                self._smiles_cache_order.remove(dp_idx)
            self._smiles_cache_order.append(dp_idx)
            return
        
        if (self.max_smiles_cache_files > 0 and 
            len(self._smiles_cache_order) >= self.max_smiles_cache_files):
            evict_idx = self._smiles_cache_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        datapoint._ensure_smiles()
        self._smiles_cache_order.append(dp_idx)
    
    def __getitem__(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        self._ensure_smiles_with_cache(dp_idx)
        tokens, features, smiles, local_idx = self.data[dp_idx][real_idx]
        return tokens, features, smiles, idx
    
    def clean_cache(self):
        for d in self.data:
            d.clean_cache()
        self._smiles_cache_order = []
    
    def shuffle(self, seed: int = None):
        pass


class KermtHybridPreTokenizedCollator(object):
    """
    Collator for hybrid training with pre-tokenized data.
    
    Used with: PreTokenizedWithFeaturesMolDataset
    Input: List of (tokens, features, smiles, idx) tuples
    Output: dict with graph_input, decoder inputs, vocab targets
    """
    
    def __init__(self, shared_dict, smiles_vocab, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.smiles_vocab = smiles_vocab
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_data):
        tokens_list, features_list, smiles_batch, idx = zip(*batch_data)
        smiles_batch = list(smiles_batch)
        idx = list(idx)
        
        # Build graph input
        batchgraph, rdkit_bond_idx = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                                        self.cmm_feature_tensors, self.cmm_feature_range)
        
        # Vocab targets
        targets = _generate_vocab_targets(smiles_batch, self.atom_vocab, self.bond_vocab,
                                          features_list=features_list,
                                          rdkit_bond_idx=rdkit_bond_idx)
        
        # Pad and prepare decoder sequences
        tokens_tensor, padding_mask = _pad_pretokenized_sequences(
            tokens_list, self.smiles_vocab.pad_index
        )
        decoder_input, decoder_target = prepare_decoder_sequences(tokens_tensor)
        decoder_padding_mask = padding_mask[:, 1:].contiguous()
        
        seq_len = decoder_input.size(1)
        positional_encoding = getattr(self.args, 'decoder_positional_encoding', 'rope')
        causal_mask = create_causal_mask(seq_len, positional_encoding)
        
        return {
            "graph_input": batchgraph,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_padding_mask": decoder_padding_mask,
            "causal_mask": causal_mask,
            "targets": targets,
            "smiles": smiles_batch,
            "idx": idx
        }


class KermtVocabPreTokenizedCollator(object):
    """
    DEPRECATED: Use KermtVocabFeaturesOnlyCollator instead.
    
    This collator receives but discards pre-tokenized SMILES (wasteful).
    Kept for backward compatibility only.
    """
    
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_data):
        # Unpack and discard tokens (wasteful but kept for API compatibility)
        _, features_list, smiles_batch, idx = zip(*batch_data)
        smiles_batch = list(smiles_batch)
        idx = list(idx)
        
        batchgraph, rdkit_bond_idx = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                                        self.cmm_feature_tensors, self.cmm_feature_range)
        targets = _generate_vocab_targets(smiles_batch, self.atom_vocab, self.bond_vocab,
                                          features_list=features_list,
                                          rdkit_bond_idx=rdkit_bond_idx)
        
        return {
            "graph_input": batchgraph,
            "targets": targets,
            "smiles": smiles_batch,
            "idx": idx
        }


# ============================================================================
# SECTION 5: Memory-Mapped Data Loading - Vocab-Only (Optimized)
# ============================================================================
# Optimized for vocab-only pretraining - doesn't load unnecessary tokens.

def get_features_only_data(
    data_path: str,
    features_mmap_dir: str,
    logger=None,
    max_smiles_cache_files: int = 50
):
    """
    Load memory-mapped features with SMILES for vocab-only pretraining.
    
    Optimized for vocab mode - does NOT load pre-tokenized SMILES.
    
    Args:
        data_path: Base data path (for graph/ SMILES and summary.txt)
        features_mmap_dir: Path to memory-mappable feature .npy files
        logger: Optional logger
        max_smiles_cache_files: Max files to keep SMILES cached
        
    Returns:
        (FeaturesOnlyMolDataset, sample_per_file)
    """
    debug = logger.debug if logger is not None else print
    
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Make sure you've run prepare_zinc15_unified.py to completion."
        )
    
    with open(summary_path) as fin:
        n_files = int(fin.readline().strip().split(":")[-1])
        n_samples = int(fin.readline().strip().split(":")[-1])
        sample_per_file = int(fin.readline().strip().split(":")[-1])
    
    debug("Loading features-only data (vocab mode):")
    debug(f"  Number of files: {n_files}")
    debug(f"  Number of samples: {n_samples}")
    debug(f"  Samples/file: {sample_per_file}")
    debug(f"  Features mmap directory: {features_mmap_dir}")
    debug(f"  Max SMILES cache files: {max_smiles_cache_files}")
    
    datapoints = []
    for i in range(n_files):
        features_path_i = os.path.join(features_mmap_dir, str(i) + ".npy")
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        if n_samples_i == 0:
            n_samples_i = sample_per_file
        
        datapoints.append(FeaturesOnlyDatapoint(
            features_file=features_path_i,
            smiles_file=smiles_path_i,
            n_samples=n_samples_i
        ))
    
    return FeaturesOnlyMolDataset(
        datapoints, max_smiles_cache_files=max_smiles_cache_files
    ), sample_per_file


class FeaturesOnlyDatapoint:
    """
    A batch datapoint with memory-mapped features only (no tokens).
    
    Optimized for vocab-only pretraining which doesn't need decoder.
    """
    
    def __init__(self, features_file: str, smiles_file: str, n_samples: int):
        self.features_file = features_file
        self.smiles_file = smiles_file
        self.n_samples = n_samples
        self._features_mmap = None
        self._smiles_cache = None
    
    def _ensure_features_mmap(self):
        if self._features_mmap is None:
            self._features_mmap = np.load(self.features_file, mmap_mode='r')
    
    def _ensure_smiles(self):
        if self._smiles_cache is None:
            self._smiles_cache = []
            with open(self.smiles_file) as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    self._smiles_cache.append(line[0])
    
    def get_features(self, idx: int) -> np.ndarray:
        self._ensure_features_mmap()
        return self._features_mmap[idx]
    
    def get_smiles(self, idx: int) -> str:
        self._ensure_smiles()
        return self._smiles_cache[idx]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.get_features(idx), self.get_smiles(idx), idx
    
    def is_loaded(self):
        return True
    
    def clean_cache(self):
        self._smiles_cache = None


class FeaturesOnlyMolDataset(Dataset):
    """
    Dataset for features-only data (optimized vocab mode).
    
    Used with: KermtVocabFeaturesOnlyCollator
    Returns: (features, smiles, idx)
    """
    
    def __init__(self, data: List[FeaturesOnlyDatapoint], graph_per_file=None,
                 max_smiles_cache_files: int = 50):
        self.data = data
        self.len = sum(len(d) for d in self.data)
        self.max_smiles_cache_files = max_smiles_cache_files
        self._smiles_cache_order = []
        
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
    
    def __len__(self) -> int:
        return self.len
    
    def _ensure_smiles_with_cache(self, dp_idx: int):
        datapoint = self.data[dp_idx]
        if datapoint._smiles_cache is not None:
            if dp_idx in self._smiles_cache_order:
                self._smiles_cache_order.remove(dp_idx)
            self._smiles_cache_order.append(dp_idx)
            return
        
        if (self.max_smiles_cache_files > 0 and 
            len(self._smiles_cache_order) >= self.max_smiles_cache_files):
            evict_idx = self._smiles_cache_order.pop(0)
            self.data[evict_idx].clean_cache()
        
        datapoint._ensure_smiles()
        self._smiles_cache_order.append(dp_idx)
    
    def __getitem__(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        self._ensure_smiles_with_cache(dp_idx)
        features, smiles, local_idx = self.data[dp_idx][real_idx]
        return features, smiles, idx
    
    def clean_cache(self):
        for d in self.data:
            d.clean_cache()
        self._smiles_cache_order = []
    
    def shuffle(self, seed: int = None):
        pass


class KermtVocabFeaturesOnlyCollator(object):
    """
    Collator for vocab-only pretraining with memory-mapped features.
    
    This is the preferred collator for vocab-only mode with large datasets.
    
    Used with: FeaturesOnlyMolDataset
    Input: List of (features, smiles, idx) tuples
    Output: dict with graph_input, vocab targets
    """
    
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.cmm_feature_tensors, self.cmm_feature_range = setup_cuik_molmaker_features(args)

    def __call__(self, batch_data):
        features_list, smiles_batch, idx = zip(*batch_data)
        smiles_batch = list(smiles_batch)
        idx = list(idx)
        
        batchgraph, rdkit_bond_idx = _build_graph_input(smiles_batch, self.shared_dict, self.args,
                                                        self.cmm_feature_tensors, self.cmm_feature_range)
        targets = _generate_vocab_targets(smiles_batch, self.atom_vocab, self.bond_vocab,
                                          features_list=features_list,
                                          rdkit_bond_idx=rdkit_bond_idx)
        
        return {
            "graph_input": batchgraph,
            "targets": targets,
            "smiles": smiles_batch,
            "idx": idx
        }
