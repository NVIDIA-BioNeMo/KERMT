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
The parsing functions for the argument input.
"""
import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory

import torch

from kermt.data.molfeaturegenerator import get_available_features_generators
from kermt.util.utils import makedirs


def add_common_args(parser: ArgumentParser):
    parser.add_argument('--no_cache', action='store_true', default=True,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--gpu', type=int, default=0,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--wandb_project', type=str,
                        help='Wandb project name. If this is provided, WandB will be used to log the training process.')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')



def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    add_common_args(parser)

    parser.add_argument('--data_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')

    parser.add_argument('--output_path', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')

    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    # NOTE: rdkit2D_normalization_type is deliberately NOT a predict argument. It is
    # baked into the features the checkpoint was finetuned on, so the only correct
    # value is the checkpoint's. make_predictions() copies it (and every other arg the
    # predict parser does not define) out of the checkpoint, which only works while the
    # attribute is absent here -- argparse setting a default would shadow it.
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--use_cuikmolmaker_featurization', action='store_true', default=False,
                        help='Use cuik-molmaker package for featurization of atoms and bonds in molecules')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for prediction')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')


def add_fingerprint_args(parser):
    add_common_args(parser)
    # parameters for fingerprints generation
    parser.add_argument('--data_path', type=str, help='Input csv file which contains SMILES')
    parser.add_argument('--output_path', type=str,
                        help='Path to npz file where predictions will be saved')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for fingerprint generation')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--fingerprint_source', type=str,
                        choices=['atom', 'bond', 'both'], default='both',
                        help='The source to generate the fingerprints.')
    parser.add_argument('--use_cuikmolmaker_featurization', action='store_true', default=False,
                        help='Use cuik-molmaker package for featurization of atoms and bonds in molecules')
    parser.add_argument('--checkpoint_path', type=str, help='model path')


def add_finetune_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """

    # General arguments
    add_common_args(parser)
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Add tensorboard logger')

    # Data argumenets
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file.')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    # Disable this option due to some bugs.
    # parser.add_argument('--test', action='store_true', default=False,
    #                     help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features.')
    parser.add_argument('--rdkit2D_normalization_type', type=str, choices=("fast", "best", "descriptastorus"), default='fast',
                        help='Type of normalization for rdkit2D features. Choices: fast, best, descriptastorus')
    parser.add_argument('--use_cuikmolmaker_featurization', action='store_true', default=False,
                        help='Use cuik-molmaker package for featurization of atoms and bonds in molecules')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator).')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')

    # Data splitting.
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.'
                             'Also used as seed for seeding everything under the sun')

    # Metric
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc',
                                 'prc-auc',
                                 'rmse',
                                 'mae',
                                 'r2',
                                 'accuracy',
                                 'recall',
                                 'sensitivity',
                                 'specificity',
                                 'matthews_corrcoef',
                                 'spearmanr'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--use_mtl_loss', action='store_true', default=False,
                        help='Use MTL loss function')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--task_wise_checkpoint', action='store_true', default=False,
                        help='Checkpoint the model for each task separately')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to task')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--early_stop_epoch', type=int, default=1000, help='If val loss did not drop in '
                                                                           'this epochs, stop running')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models for ensemble prediction.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    
    # Encoder architecture arguments (required when training from scratch without checkpoint)
    parser.add_argument('--hidden_size', type=int, default=800,
                        help='Encoder hidden dimension. Default: 800 (matches pretrained models)')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of encoder message passing layers. Default: 6')
    parser.add_argument('--num_attn_head', type=int, default=4,
                        help='Number of attention heads in encoder MTBlock. Default: 4')
    parser.add_argument('--num_mt_block', type=int, default=1,
                        help='Number of MTBlocks in encoder. Default: 1')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to encoder linear layers. Default: False')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Use undirected edges (sum the two relevant bond vectors). Default: False')
    
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--ffn_task_specific_hidden_size', type=int, default=None,
                        help='Hidden size for task-specific FFN layers (and common FFN '
                             'output when task-specific layers are used). Required if '
                             'ffn_num_task_specific_layers > 0.')
    parser.add_argument('--ffn_num_task_specific_layers', type=int, default=0,
                        help='Number of task-specific layers in FFN after common FFN layer encoding')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--select_by_loss', action='store_true', default=False,
                        help='Use validation loss as refence standard to select best model to predict')

    parser.add_argument("--embedding_output_type", default="atom", choices=["atom", "bond", "both"],
                        help="This the model parameters for pretrain model. The current finetuning task only use the "
                             "embeddings from atom branch. ")

    # Self-attentive readout.
    parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer. '
                                                                                     'Otherwise use mean aggregation '
                                                                                     'layer.')
    parser.add_argument('--attn_hidden', type=int, default=4, nargs='?', help='Self attention layer '
                                                                              'hidden layer size.')
    parser.add_argument('--attn_out', type=int, default=8, nargs='?', help='Self attention layer '
                                                                             'output feature size (FFN input = hidden_size * attn_out).')

    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for output of two branches.')


    parser.add_argument('--bond_drop_rate', type=float, default=0, help='Drop out bond in molecular.')
    parser.add_argument('--distinct_init', action='store_true', default=False,
                        help='Using distinct weight init for model ensemble')
    parser.add_argument('--fine_tune_coff', type=float, default=1.0,
                        help='Enable distinct fine tune learning rate for fc and other layer')

    # For multi-gpu finetune.
    parser.add_argument('--enbl_multi_gpu', dest='enbl_multi_gpu',
                        action='store_true', default=False,
                        help='enable multi-GPU training')
    
    # Add hyperparameter optimization
    parser.add_argument("--n_trials", type=int, dest="n_trials",
                        help="Number of trials for hyperparameter optimization using Optuna")
    parser.add_argument("--hpo_mode", type=str, default="all", choices=["all", "openadmet"],
                        help="HPO mode: 'all' for full model tuning (default), "
                             "'openadmet' for small dataset with frozen encoder and 1 FFN layer")


def add_pretrain_args(parser: ArgumentParser):
    """
    Add arguments for pretraining with organized parameter groups.
    Supports both vocabulary-based pretraining and CMIM pretraining with decoder.
    """
    
    # ========== System and Hardware Arguments ==========
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Enable gpu training or not.')
    parser.add_argument('--enable_multi_gpu', dest='enable_multi_gpu',
                        action='store_true', default=False,
                        help='Enable multi-GPU training')
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for pretraining. Unset (the default) leaves every "
                             "RNG untouched, which is how pretraining has always run -- two "
                             "runs of the same command give different results. Set it to make "
                             "a run reproducible: model init, dropout, the dyMPN depth draw, "
                             "the vocab masking and any bond dropout are all seeded from it. "
                             "Needed for A/B comparisons, where an unseeded run-to-run spread "
                             "can otherwise exceed the effect being measured. Checkpoints carry "
                             "no RNG state, so a run resumed from one is not bit-identical to an "
                             "uninterrupted seeded run.")
    
    # ========== Data Arguments ==========
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to train data CSV file')
    parser.add_argument('--val_data_path', type=str, required=False,
                        help='Path to val data CSV file')
    parser.add_argument('--test_data_path', type=str, required=False, default=None,
                        help='Path to test data CSV file')
    parser.add_argument('--lazy_loading', action='store_true', default=False,
                        help='Skip pre-loading all data files. Use for very large datasets (e.g., full ZINC15). '
                             'Data will be loaded on-demand during training with LRU cache eviction.')
    parser.add_argument('--max_cached_files', type=int, default=100,
                        help='Maximum number of data files to keep in memory (LRU cache). '
                             'Used for lazy_loading mode and memory-mapped SMILES cache. '
                             'For large datasets (e.g., ZINC15 200M with 417 files), set to 500+ '
                             'to cache all SMILES and avoid CSV parsing during training. '
                             'Default: 100 files.')
    
    # Vocabulary paths (mode-dependent)
    parser.add_argument('--atom_vocab_path', type=str, required=False, default=None,
                        help="Path to atom vocabulary (required for vocab-based pretraining).")
    parser.add_argument('--bond_vocab_path', type=str, required=False, default=None,
                        help="Path to bond vocabulary (required for vocab-based pretraining).")
    parser.add_argument('--smiles_vocab_path', type=str, required=False, default=None,
                        help="Path to SMILES vocabulary (required for CMIM/decoder training).")
    
    # Pre-tokenized data for memory-efficient CMIM training
    parser.add_argument('--tokens_dir', type=str, required=False, default=None,
                        help="Path to pre-tokenized .npy files for CMIM training. "
                             "When provided, uses memory-mapped loading for efficient multi-worker training. "
                             "Generated by pretokenize_zinc15.py or prepare_zinc15_unified.py script.")
    parser.add_argument('--features_mmap_dir', type=str, required=False, default=None,
                        help="Path to memory-mappable feature .npy files for hybrid/vocab training. "
                             "When provided with --tokens_dir, enables multi-worker data loading for "
                             "hybrid and vocab training modes. Generated by prepare_zinc15_unified.py "
                             "(feature_mmap/ directory).")
    parser.add_argument('--fg_label_path', type=str, nargs='*',
                        help='Path to functional group task labels (optional).')
    
    # Featurization
    parser.add_argument('--use_cuikmolmaker_featurization', action='store_true', default=False,
                        help='Use cuik-molmaker package for molecule featurization')
    
    # ========== Training Mode Selection ==========
    parser.add_argument('--pretrain_mode', type=str, default='vocab',
                        choices=['vocab', 'cmim', 'hybrid'],
                        help='Pretraining mode: '
                             '"vocab" = original GROVER vocabulary prediction (av/bv/fg tasks), '
                             '"cmim" = CMIM contrastive learning + SMILES reconstruction, '
                             '"hybrid" = both CMIM and vocab objectives combined. '
                             'Default: vocab')
    parser.add_argument('--vocab_loss_weight', type=float, default=1.0,
                        help='Weight for vocabulary prediction loss in hybrid training. '
                             'Total loss = CMIM_loss + vocab_loss_weight * vocab_loss. Default: 1.0')
    
    # ========== Encoder Model Arguments ==========
    parser.add_argument("--backbone", default="gtrans", choices=["gtrans", "dualtrans"],
                        help="Encoder backbone architecture. `dualtrans` is the legacy "
                             "name for the same architecture, kept for compatibility with "
                             "older grover_base checkpoints.")
    parser.add_argument('--embedding_output_type', type=str, default='both', nargs='?',
                        choices=("atom", "bond", "both"),
                        help="Type of output embeddings from encoder. Options: atom, bond, both")
    parser.add_argument('--hidden_size', type=float, default=800,
                        help='Encoder hidden dimension. Must be divisible by --num_attn_head. '
                             'Default: 800, matching the released checkpoints and the '
                             'finetune parser. (The previous default of 3 came with help text '
                             'claiming a x100 scaling that was never implemented, so it built '
                             'a 3-dimensional encoder.)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of encoder message passing layers.')
    parser.add_argument('--num_attn_head', type=int, default=4, 
                        help='Number of attention heads in encoder MTBlock.')
    parser.add_argument('--num_mt_block', type=int, default=1, 
                        help="Number of MTBlocks in encoder.")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability for encoder.')
    parser.add_argument('--activation', type=str, default='PReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function for encoder.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to encoder linear layers')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Use undirected edges (sum the two relevant bond vectors)')
    parser.add_argument('--bond_drop_rate', type=float, default=0, 
                        help='Dropout rate for bonds in molecular graph')
    parser.add_argument('--dist_coff', type=float, default=0.1, 
                        help='Disagreement coefficient for atom and bond branches.')
    
    # Readout layer (for molecule-level embedding)
    parser.add_argument('--self_attention', action='store_true', default=False,
                        help='Use self-attention readout layer. '
                             'Default: False (uses mean aggregation).')
    parser.add_argument('--attn_hidden', type=int, default=4,
                        help='Self-attention readout hidden layer size.')
    parser.add_argument('--attn_out', type=int, default=8,
                        help='Self-attention readout output feature size (FFN input = hidden_size * attn_out).')
    
    # ========== CMIM-Specific Arguments ==========
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Dimension of latent space for CMIM. This also determines decoder hidden size '
                             '(latent_dim must equal decoder hidden_size for cross-attention). '
                             'Default: 512.')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='Temperature parameter for contrastive loss in CMIM. '
                             'Default: 0.1')
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0,
                        help='Weight for SMILES reconstruction loss in CMIM training. '
                             'Default: 1.0')
    parser.add_argument('--normalize_gradient', action='store_true', default=False,
                        help='Normalize gradients of CMIM loss components (log_q_z_given_x, log_P_z) by latent dimensionality. '
                             'Useful when latent_dim is large and dominates gradient magnitudes. '
                             'Default: False')
    parser.add_argument('--normalize_loss', action='store_true', default=False,
                        help='Normalize CMIM loss values by latent dimensionality. '
                             'Scales the loss values themselves (affects what gets logged). '
                             'Can be used independently or together with --normalize_gradient. '
                             'Default: False')
    
    # ========== Decoder Model Arguments (for CMIM with reconstruction) ==========
    parser.add_argument('--decoder_num_layers', type=int, default=3,
                        help='Number of transformer decoder layers. Default: 3')
    parser.add_argument('--decoder_num_attention_heads', type=int, default=8,
                        help='Number of attention heads in decoder. Default: 8')
    parser.add_argument('--decoder_ffn_hidden_size', type=int, default=2048,
                        help='Decoder feedforward hidden size. '
                             'Default: 2048 (4 * latent_dim with latent_dim=512).')
    parser.add_argument('--decoder_dropout', type=float, default=0.1,
                        help='Dropout probability for decoder. Default: 0.1')
    parser.add_argument('--decoder_max_seq_len', type=int, default=512,
                        help='Maximum SMILES sequence length for decoder (sequences will be truncated). '
                             'Default: 512. Increase if you have longer SMILES.')
    parser.add_argument('--decoder_positional_encoding', type=str, default='rope',
                        choices=['rope', 'sinusoidal'],
                        help='Type of positional encoding for decoder. '
                             'Options: "rope" (Rotary Position Embedding) or "sinusoidal" (classic additive). '
                             'Default: rope')
    parser.add_argument('--decoder_gate_self_attn', action='store_true', default=False,
                        help='Enable G1 gating for decoder self-attention layers. '
                             'Applies multiplicative sigmoid gating after SDPA output. '
                             'Only works with positional_encoding="rope". Default: False')
    parser.add_argument('--decoder_gate_cross_attn', action='store_true', default=False,
                        help='Enable G1 gating for decoder cross-attention layers. '
                             'Applies multiplicative sigmoid gating after SDPA output. '
                             'Only works with positional_encoding="rope". Default: False')
    
    # ========== Training Arguments ==========
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU (micro batch size)')
    parser.add_argument('--num_dataloader_workers', type=int, default=0,
                        help='Number of workers for dataloader')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of warmup epochs for learning rate schedule')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay for optimizer')
    parser.add_argument('--max_val_batches', type=int, default=10,
                        help='Maximum number of batches in validation loop.')
    parser.add_argument('--val_interval', type=int, default=0,
                        help='Run validation every N steps (0 = only at end of each epoch). '
                             'Similar to save_interval; e.g. 500 to validate every 500 steps.')
    
    # ========== Checkpoint and Logging Arguments ==========
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_interval', type=int, default=100, 
                        help='Model saving interval (in steps).')
    parser.add_argument("--tensorboard", action="store_true", default=False,
                        help="Use tensorboard to visualize training.")
    parser.add_argument('--train_interval', type=int, default=10,
                        help='Log train metrics to TensorBoard every N steps (default: 10).')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Wandb project name. If provided, WandB will be used to log pretraining metrics.')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')

def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return
    if not hasattr(args, 'checkpoint_path'):
        args.checkpoint_path = None

    if not hasattr(args, 'checkpoint_dir'):
        args.checkpoint_dir = None

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    if args.parser_name == "eval":
        assert args.ensemble_size * args.num_folds == len(args.checkpoint_paths)

    args.ensemble_size = len(args.checkpoint_paths)



    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.data_path
    assert args.output_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    makedirs(args.output_path, isfile=True)
    setattr(args, 'fingerprint', False)


def modify_fingerprint_args(args):
    assert args.data_path
    assert args.output_path
    assert args.checkpoint_path is not None or args.checkpoint_paths is not None


    update_checkpoint_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    makedirs(args.output_path, isfile=True)
    setattr(args, 'fingerprint', True)


def get_newest_train_args():
    """
    For backward compatibility.

    :return:  A Namespace containing the newest training arguments
    """
    dummy_parser = ArgumentParser()
    add_finetune_args(dummy_parser)
    args = dummy_parser.parse_args(args=[])
    args.data_path = ''
    modify_train_args(args)
    return args


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global TEMP_DIR  # Prevents the temporary directory from being deleted upon function return

    assert args.data_path is not None
    assert args.dataset_type is not None

    if args.save_dir is not None:
        makedirs(args.save_dir)
    else:
        TEMP_DIR = TemporaryDirectory()
        args.save_dir = TEMP_DIR.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'r2', "spearmanr"])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae']

    update_checkpoint_args(args)

    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    args.num_lrs = 1



    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)
    assert (args.split_type == 'crossval') == (args.crossval_index_dir is not None)
    assert (args.split_type in ['crossval', 'index_predetermined']) == (args.crossval_index_file is not None)
    if args.split_type in ['crossval', 'index_predetermined']:
        # Default is JSON (secure). Explicit ``.pkl`` / ``.pckl`` opts into pickle
        # so older on-disk crossval index files can still be loaded if encountered.
        if args.crossval_index_file.endswith(('.pkl', '.pckl')):
            with open(args.crossval_index_file, 'rb') as rf:
                args.crossval_index_sets = pickle.load(rf)
        else:
            with open(args.crossval_index_file, 'r') as rf:
                args.crossval_index_sets = json.load(rf)
        args.num_folds = len(args.crossval_index_sets)
        args.seed = 0


    if args.bond_drop_rate > 0:
        args.no_cache = True

    setattr(args, 'fingerprint', False)
    
    # Set dense=False for encoder (required when training from scratch)
    args.dense = False

    # Set dense=False for encoder (required when training from scratch)
    args.dense = False


def modify_pretrain_args(args: Namespace):
    """

    :param args:
    :return:
    """
    args.dense = False
    args.fine_tune_coff = 1
    args.no_cache = True
    args.hidden_size = int(args.hidden_size)

def parse_args_ddp() -> Namespace:
    """Parse arguments for DDP training"""
    parser = ArgumentParser()
    add_pretrain_args(parser)
    args = parser.parse_args()

    modify_pretrain_args(args)
    return args

def parse_args() -> Namespace:
    """
    Parses arguments for training and testing (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    subparser = parser.add_subparsers(title="subcommands",
                                      dest="parser_name",
                                      help="Subcommands for fintune, prediction, and fingerprint.")
    parser_finetune = subparser.add_parser('finetune', help="Fine tune the pre-trained model.")
    add_finetune_args(parser_finetune)
    parser_eval = subparser.add_parser('eval', help="Evaluate the results of the pre-trained model.")
    add_finetune_args(parser_eval)
    parser_predict = subparser.add_parser('predict', help="Predict results from fine tuned model.")
    add_predict_args(parser_predict)
    parser_fp = subparser.add_parser('fingerprint', help="Get the fingerprints of SMILES.")
    add_fingerprint_args(parser_fp)
    parser_pretrain = subparser.add_parser('pretrain', help="Pretrain with unlabelled SMILES.")
    add_pretrain_args(parser_pretrain)

    args = parser.parse_args()

    if args.parser_name == 'finetune' or args.parser_name == 'eval':
        modify_train_args(args)
    elif args.parser_name == "pretrain":
        modify_pretrain_args(args)
    elif args.parser_name == 'predict':
        modify_predict_args(args)
    elif args.parser_name == 'fingerprint':
        modify_fingerprint_args(args)

    return args
