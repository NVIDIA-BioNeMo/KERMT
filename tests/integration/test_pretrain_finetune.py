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
import pytest


import subprocess
import os
import shutil

def pretrain_ddp(data_dir, world_size):
    # Set up environment variable for WORLD_SIZE
    env = os.environ.copy()
    env["WORLD_SIZE"] = str(world_size)

    # Build the command as a list for subprocess
    cmd = [
        "python", "pretrain_ddp.py",
        "--train_data_path", str(data_dir / "pretrain/train_9k"),
        "--val_data_path", str(data_dir / "pretrain/val_1k"),
        "--save_dir", f"test_run/pretrain/model/train_val_ws{world_size}",
        "--atom_vocab_path", str(data_dir / "pretrain/pretrain_atom_vocab.json"),
        "--bond_vocab_path", str(data_dir / "pretrain/pretrain_bond_vocab.json"),
        "--batch_size", "32",
        "--dropout", "0.1",
        "--depth", "6",
        "--num_attn_head", "4",
        "--hidden_size", "800",
        "--epochs", "2",
        "--init_lr", "1E-5",
        "--max_lr", "1.5E-4",
        "--final_lr", "1E-5",
        "--warmup_epochs", "20",
        "--weight_decay", "1E-7",
        "--activation", "PReLU",
        "--backbone", "gtrans",
        "--embedding_output_type", "both",
        "--tensorboard",
        "--save_interval", "100",
    ]

    # Run the command and check for successful execution
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print stdout and stderr for debugging if needed
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Assert that the process exited successfully
    assert result.returncode == 0, f"Process failed with exit code {result.returncode}"



def finetune(data_dir):

    cmd = [
        "python", "main.py", "finetune",
        "--data_path", str(data_dir / "finetune/train.csv"),
        "--separate_val_path", str(data_dir / "finetune/val.csv"),
        "--separate_test_path", str(data_dir / "finetune/test.csv"),
        "--save_dir", "test_run/finetune",
        "--checkpoint_path", "test_run/pretrain/model/train_val_ws1/last_checkpoint.pt",
        "--dataset_type", "regression",
        "--split_type", "scaffold_balanced",
        "--ensemble_size", "1",
        "--num_folds", "1",
        "--no_features_scaling",
        "--ffn_hidden_size", "700",
        "--ffn_num_layers", "3",
        "--bond_drop_rate", "0.1",
        "--epochs", "2",
        "--metric", "mae",
        "--self_attention",
        "--dist_coff", "0.15",
        "--max_lr", "1e-4",
        "--final_lr", "2e-5",
        "--dropout", "0.0",
        "--use_cuikmolmaker_featurization",
        "--features_generator", "rdkit_2d_normalized_cuik_molmaker",
        "--rdkit2D_normalization_type", "descriptastorus",
        "--seed", "50"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Finetune command failed with exit code {result.returncode}"


def predict(data_dir):
    # Make output dir
    os.makedirs("test_run/predict", exist_ok=True)
    cmd = [
        "python", "main.py", "predict",
        "--data_path", str(data_dir / "finetune/test.csv"),
        "--checkpoint_dir", "test_run/finetune/",
        "--no_features_scaling",
        "--features_generator", "rdkit_2d_normalized_cuik_molmaker",
        "--output", "test_run/predict/predict.csv"
    ]
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"Predict command failed with exit code {result.returncode}"
    assert os.path.exists("test_run/predict/predict.csv"), "Predict command failed to create predict.csv"


def test_pretrain_ddp_finetune(data_dir):
    import torch
    os.makedirs("test_run", exist_ok=True)

    world_size = 1
    pretrain_ddp(data_dir, world_size)

    # Check that the checkpoint file exists
    assert os.path.exists("test_run/pretrain/model/train_val_ws1/last_checkpoint.pt"), "Checkpoint file from pretrain not found"

    print(f"Pre-trained with world size {world_size} completed")
    finetune(data_dir)
    print(f"Finetuned with world size {world_size} completed")

    predict(data_dir)
    print("Prediction completed")

    # Only test multi-GPU if enough GPUs are available
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        world_size = 2
        pretrain_ddp(data_dir, world_size)
        print(f"Pre-trained with world size {world_size} completed")
    else:
        print(f"Skipping multi-GPU test (only {num_gpus} GPU(s) available)")

    # Cleanup 
    if os.path.exists("test_run"):
        shutil.rmtree("test_run")