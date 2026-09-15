# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint/vocabulary compatibility and artifact-boundary regression tests."""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
import importlib.util
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "skills/_shared/scripts/_utils.py"
spec = importlib.util.spec_from_file_location("kermt_artifact_utils", SCRIPT)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class _UntrustedPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        # A regression to unrestricted loading would create only this test's
        # marker, never execute any real workload or access credentials.
        return os.system, ("touch " + shlex.quote(str(self.marker)),)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_checkpoint_preserves_model_resume_state_and_numpy_scalers(tmp_path, dtype):
    weights = torch.arange(4, dtype=torch.float32)
    scaler = np.array([0, 1, 2], dtype=dtype)
    path = tmp_path / "checkpoint.pt"
    torch.save({
        "args": Namespace(hidden_size=800, depth=6),
        "state_dict": {"kermt.encoder.weight": weights},
        "data_scaler": {"means": scaler, "stds": scaler + 1},
        "features_scaler": None,
        "optimizer": {"state": {0: {"momentum_buffer": weights + 1}}, "param_groups": []},
        "scheduler_step": 17,
        "epoch": 2,
        "batch_idx": 4,
        "wandb_run_id": "resume-fixture",
        "best_score": dtype(1),
    }, path)

    loaded = utils.load_checkpoint(path)
    assert loaded["args"] == Namespace(hidden_size=800, depth=6)
    torch.testing.assert_close(loaded["state_dict"]["kermt.encoder.weight"], weights)
    np.testing.assert_array_equal(loaded["data_scaler"]["means"], scaler)
    assert loaded["data_scaler"]["means"].dtype == scaler.dtype
    torch.testing.assert_close(loaded["optimizer"]["state"][0]["momentum_buffer"], weights + 1)
    assert (loaded["scheduler_step"], loaded["epoch"], loaded["batch_idx"]) == (17, 2, 4)
    assert loaded["wandb_run_id"] == "resume-fixture"
    assert loaded["best_score"] == dtype(1)


def test_checkpoint_rejects_executable_pickle_without_creating_marker(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "checkpoint.pt"
    torch.save({"args": Namespace(), "state_dict": {}, "extra": _UntrustedPayload(marker)}, path)
    with pytest.raises(pickle.UnpicklingError):
        utils.load_checkpoint(path)
    assert not marker.exists()


@pytest.mark.parametrize("vocab_class_name", ["MolVocab", "SMILESVocab"])
def test_existing_vocabulary_objects_keep_their_token_counts(tmp_path, vocab_class_name):
    from kermt.data import torchvocab

    vocab = object.__new__(getattr(torchvocab, vocab_class_name))
    vocab.stoi = {"<pad>": 0, "C": 1, "O": 2}
    vocab.itos = list(vocab.stoi)
    vocab.freqs = Counter({"C": 4, "O": 2})
    vocab.regex = re.compile(r"C|O")
    path = tmp_path / "vocab.pkl"
    path.write_bytes(pickle.dumps(vocab))
    assert utils.count_vocab_entries(path) == 3


def test_legacy_grover_vocab_is_counted_without_importing_grover(tmp_path):
    # Protocol-2 state from the original GROVER module name. The reader should
    # use only its stored token mapping, without importing that runtime package.
    data = (b"\x80\x02cgrover.data.torchvocab\nMolVocab\n)\x81}"
            b"X\x04\x00\x00\x00stoi}X\x01\x00\x00\x00CK\x00ssb.")
    path = tmp_path / "grover.pkl"
    path.write_bytes(data)
    assert utils.count_vocab_entries(path) == 1


def test_vocabulary_rejects_executable_pickle_without_creating_marker(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "vocab.pkl"
    path.write_bytes(pickle.dumps(_UntrustedPayload(marker)))
    with pytest.raises(ValueError, match="unsupported vocabulary object"):
        utils.validate_vocab_file(path, kind="smiles")
    assert not marker.exists()


def test_job_environment_keeps_runtime_settings_and_excludes_unrelated_secrets(tmp_path):
    source = {
        "PATH": str(Path(sys.executable).parent), "CUDA_VISIBLE_DEVICES": "3,5",
        "NCCL_DEBUG": "INFO", "OMP_NUM_THREADS": "1", "PYTHONPATH": str(tmp_path),
        "HF_TOKEN": "test-hf-secret", "UNRELATED_SECRET": "test-unrelated-secret",
        "NVIDIA_INFERENCE_KEY": "test-inference-secret", "WANDB_API_KEY": "test-wandb-secret",
    }
    with patch.dict(os.environ, source, clear=True):
        env = utils.runner_environment(REPO_ROOT)
        assert env["CUDA_VISIBLE_DEVICES"] == "3,5"
        assert env["NCCL_DEBUG"] == "INFO"
        assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO_ROOT), str(tmp_path)]
        for key in ("HF_TOKEN", "UNRELATED_SECRET", "NVIDIA_INFERENCE_KEY", "WANDB_API_KEY"):
            assert key not in env
        opted_in = utils.runner_environment(REPO_ROOT, wandb=True)
        assert opted_in["WANDB_API_KEY"] == "test-wandb-secret"
        assert "HF_TOKEN" not in opted_in

    proc = subprocess.run(
        [sys.executable, "-c", "import os; assert os.environ['CUDA_VISIBLE_DEVICES'] == '3,5'; "
         "assert os.environ['NCCL_DEBUG'] == 'INFO'; assert 'UNRELATED_SECRET' not in os.environ"],
        env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
