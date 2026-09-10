#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy canonical helpers/defaults into the skills that consume them.

Edit files under skills/_shared/, then run this script with --write and commit
the per-skill copies. CI uses --check to catch drift. Copies are real files:
nv-carps signs each skill directory independently and rejects symlinks.
"""
from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

SHARED = Path(__file__).resolve().parent
SKILLS_ROOT = SHARED.parent
PRETRAIN_SKILLS = frozenset({
    "kermt-continue-pretrain", "kermt-pretrain-scratch", "kermt-add-cmim-pretrain",
})
WORKFLOW_SKILLS = PRETRAIN_SKILLS | {"kermt-finetune", "kermt-infer", "kermt-embed"}
DOWNLOAD_SKILLS = frozenset({"kermt-continue-pretrain", "kermt-finetune", "kermt-embed"})

# Explicit consumers keep each independently installed skill complete without
# copying unrelated runners or configurations into it.
SHARED_ASSET_OWNERS = {
    "scripts/kermt_container.sh": WORKFLOW_SKILLS | {"kermt-setup"},
    "scripts/_utils.py": WORKFLOW_SKILLS,
    "scripts/check_checkpoint.py": WORKFLOW_SKILLS,
    "scripts/check_data.py": WORKFLOW_SKILLS,
    "scripts/prepare_data.py": WORKFLOW_SKILLS,
    "scripts/fetch_released_model.py": DOWNLOAD_SKILLS,
    "scripts/run_pretrain_local.py": PRETRAIN_SKILLS,
    "scripts/run_finetune_local.py": {"kermt-finetune"},
    "scripts/run_inference.py": {"kermt-infer"},
    "scripts/run_extract_embeddings.py": {"kermt-embed"},
    "scripts/upgrade_to_hybrid.py": {"kermt-add-cmim-pretrain"},
    "config/defaults_pretrain.json": PRETRAIN_SKILLS,
    "config/defaults_finetune.json": {"kermt-finetune"},
    "config/defaults_inference.json": {"kermt-infer"},
    "config/defaults_embed.json": {"kermt-embed"},
    "config/released_model.json": DOWNLOAD_SKILLS,
    "references/released-models.md": DOWNLOAD_SKILLS,
}


def sync(*, write: bool = False) -> int:
    skills = sorted(path.parent for path in SKILLS_ROOT.glob("kermt-*/SKILL.md"))
    names = {path.name for path in skills}
    problems = []
    for rel, owners in SHARED_ASSET_OWNERS.items():
        source = SHARED / rel
        if source.is_symlink() or not source.is_file():
            problems.append(f"missing canonical real file: {rel}")
        for owner in sorted(owners - names):
            problems.append(f"unknown owner {owner}: {rel}")
    for directory in ("scripts", "config", "references"):
        for source in (SHARED / directory).rglob("*"):
            if "__pycache__" in source.parts or source.suffix == ".pyc":
                continue
            if source.is_file() and source.relative_to(SHARED).as_posix() not in SHARED_ASSET_OWNERS:
                problems.append(f"canonical file has no ownership entry: {source.relative_to(SHARED)}")
    if problems:
        print("\n".join(problems))
        return 1

    for skill in skills:
        for rel, owners in SHARED_ASSET_OWNERS.items():
            source, dest = SHARED / rel, skill / rel
            if skill.name not in owners:
                if dest.exists() or dest.is_symlink():
                    if write:
                        dest.unlink()
                    else:
                        problems.append(f"{skill.name}: unexpected managed file {rel}")
                continue
            if write:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink():
                    dest.unlink()
                shutil.copy2(source, dest)
            elif dest.is_symlink() or not dest.is_file():
                problems.append(f"{skill.name}: missing real copy of {rel}")
            elif source.read_bytes() != dest.read_bytes():
                problems.append(f"{skill.name}: {rel} differs from _shared/{rel}")
            elif stat.S_IMODE(source.stat().st_mode) != stat.S_IMODE(dest.stat().st_mode):
                problems.append(f"{skill.name}: {rel} has different permissions")
    if problems:
        print("\n".join(problems))
        print("Run python3 skills/_shared/sync_shared.py --write and commit the copies.")
        return 1
    print(f"Shared assets {'synced' if write else 'verified'} for {len(skills)} skills.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="refresh per-skill copies")
    mode.add_argument("--check", action="store_true", help="fail on missing copies or drift")
    args = parser.parse_args()
    return sync(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
