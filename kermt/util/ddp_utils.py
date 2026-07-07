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
Shared DistributedDataParallel (DDP) helpers for KERMT.

Used by the pretraining and finetuning DDP launchers so both share a single,
proven single-node DDP bootstrap. The logic here matches what pretraining has
used in production (localhost rendezvous, NCCL backend, per-process GPU pinning
by rank, and topology-aware NCCL P2P configuration).
"""
import os
import subprocess

import torch
from torch.distributed import init_process_group


def configure_nccl_for_topology():
    """
    Auto-configure NCCL settings based on GPU topology.
    This handles cases where P2P (peer-to-peer) GPU communication is not available.
    Must be called BEFORE spawning processes (in main process).
    """
    # Check if user has already set NCCL settings (don't override)
    if "NCCL_P2P_DISABLE" in os.environ:
        print(f"[INFO] Using user-provided NCCL settings: NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}")
        return

    # Try to detect GPU topology
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'],
                                capture_output=True, text=True, timeout=5)
        topo_output = result.stdout

        # Check for poor GPU connectivity (SYS or NODE topology)
        # These topologies typically don't support P2P well
        if 'SYS' in topo_output or 'NODE' in topo_output:
            print("[INFO] Detected cross-NUMA or system-level GPU topology (SYS/NODE).")
            print("[INFO] Disabling P2P for stability. This is normal for multi-socket systems.")
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["NCCL_SHM_DISABLE"] = "0"
        else:
            print("[INFO] GPU topology appears to support P2P. Enabling P2P communication.")
    except Exception as e:
        # If detection fails, use safe defaults (disable P2P)
        print(f"[WARNING] Could not detect GPU topology: {e}")
        print("[INFO] Using safe default: P2P disabled. Set NCCL_P2P_DISABLE=0 to enable if your system supports it.")
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "0"


def ddp_setup(rank: int, world_size: int):
    """
    Initialize the process group for single-node DDP and pin this process to its GPU.

    Args:
        rank: Unique identifier of each process (also the GPU index it is pinned to).
        world_size: Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
