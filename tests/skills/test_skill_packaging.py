# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaging checks that need only Python and Bash, with no GPU or Docker.

Run with python3 -m unittest discover -s agent/tests -p test_skill_packaging.py.
The container tests record Docker arguments using a fake executable; they do
not start containers or exercise model computation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS = REPO_ROOT / "skills"
RUNNERS = {
    "kermt-continue-pretrain": "run_pretrain_local.py",
    "kermt-pretrain-scratch": "run_pretrain_local.py",
    "kermt-add-cmim-pretrain": "run_pretrain_local.py",
    "kermt-finetune": "run_finetune_local.py",
    "kermt-infer": "run_inference.py",
    "kermt-embed": "run_extract_embeddings.py",
}


class SkillPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kermt-packaging-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runtime(self):
        runtime = self.root / "runtime checkout"
        runtime.mkdir(exist_ok=True)
        (runtime / "kermt").mkdir(exist_ok=True)
        (runtime / "main.py").touch()
        return runtime

    def installed_skill(self, name):
        return Path(shutil.copytree(SKILLS / name, self.root / "installed skills" / name))

    def run_python(self, *args, env=None):
        return subprocess.run(
            [sys.executable, "-B", *map(str, args)], cwd=self.root,
            env=env, text=True, capture_output=True,
        )

    def test_bundles_contain_referenced_assets_and_no_symlinks(self):
        manifests = sorted(SKILLS.glob("kermt-*/SKILL.md"))
        self.assertEqual(len(manifests), 8)
        self.assertEqual((REPO_ROOT / ".claude/skills").resolve(), SKILLS)
        asset_pattern = re.compile(
            r"(?<![\w-])((?:scripts|config|references)/[\w.-]+\.(?:py|sh|json|md))"
        )
        for manifest in manifests:
            with self.subTest(skill=manifest.parent.name):
                installed = self.installed_skill(manifest.parent.name)
                for path in installed.rglob("*"):
                    self.assertFalse(path.is_symlink(), path)
                texts = [installed / "SKILL.md", installed / "skill-card.md"]
                texts.extend(installed.glob("evals/*.json"))
                for path in texts:
                    for asset in asset_pattern.findall(path.read_text()):
                        self.assertTrue((installed / asset).is_file(), f"{path}: {asset}")

    def test_installed_runners_use_the_selected_checkout_and_local_defaults(self):
        runtime = self.runtime()
        env = {**os.environ, "KERMT_REPO": str(runtime), "PYTHONDONTWRITEBYTECODE": "1"}
        probe = """
import json, runpy, sys
from pathlib import Path
module = runpy.run_path(sys.argv[1])
defaults = module['DEFAULTS_PATH']
json.loads(defaults.read_text())
print(json.dumps({
    'repo': str(module['REPO_ROOT']),
    'defaults': str(defaults),
    'validator': str(module['CHECK_CHECKPOINT_PATH']),
}))
"""
        for name, runner in RUNNERS.items():
            with self.subTest(skill=name):
                installed = self.installed_skill(name)
                proc = self.run_python("-c", probe, installed / "scripts" / runner, env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                payload = json.loads(proc.stdout)
                self.assertEqual(Path(payload["repo"]), runtime)
                self.assertEqual(Path(payload["defaults"]).parent, installed / "config")
                self.assertEqual(
                    Path(payload["validator"]), installed / "scripts/check_checkpoint.py",
                )
                self.assertTrue(Path(payload["validator"]).is_file())

    def test_explicit_invalid_checkout_does_not_fall_back_to_another_repo(self):
        installed = self.installed_skill("kermt-infer")
        env = {**os.environ, "KERMT_REPO": str(self.root / "missing")}
        proc = self.run_python(
            "-c", "import runpy, sys; runpy.run_path(sys.argv[1])['resolve_kermt_repo']()",
            installed / "scripts/_utils.py", env=env,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Set KERMT_REPO", proc.stderr)

    def test_container_mounts_include_the_independently_installed_skill(self):
        runtime = self.runtime()
        installed = self.installed_skill("kermt-infer")
        binaries = self.root / "bin"
        binaries.mkdir()
        docker = binaries / "docker"
        docker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['DOCKER_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "print('fake-container')\n"
        )
        docker.chmod(0o755)
        log = self.root / "docker.jsonl"
        # Supply only test inputs: the fake executable never receives real
        # credentials that the production helper can forward to containers.
        env = {
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "KERMT_REPO": str(runtime), "DOCKER_LOG": str(log),
        }
        for mode in ("run", "run_detached"):
            with self.subTest(mode=mode):
                proc = subprocess.run(
                    ["bash", str(installed / "scripts/kermt_container.sh"), mode,
                     "--", "python /skill/scripts/run_inference.py --help"],
                    cwd=self.root, env=env, text=True, capture_output=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        runs = [args for args in calls if args[0] == "run"]
        self.assertEqual(len(runs), 2)
        for args in runs:
            mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "-v"]
            self.assertIn(f"{runtime}:/workspace", mounts)
            self.assertIn(f"{installed}:/skill:ro", mounts)
            self.assertIn("KERMT_REPO=/workspace", args)
            self.assertEqual(args[-1], "python /skill/scripts/run_inference.py --help")

    def test_shared_edits_propagate_and_replace_symlinks_safely(self):
        copied = Path(shutil.copytree(SKILLS, self.root / "skills"))
        sync = copied / "_shared/sync_shared.py"
        original_check = self.run_python(sync, "--check")
        self.assertEqual(original_check.returncode, 0, original_check.stdout)
        source = copied / "_shared/scripts/_utils.py"
        source.write_text(source.read_text() + "\n# Packaging test fixture.\n")
        outside = self.root / "unrelated.txt"
        outside.write_text("keep this file")
        alias = copied / "kermt-infer/scripts/_utils.py"
        alias.unlink()
        alias.symlink_to(outside)
        self.assertNotEqual(self.run_python(sync, "--check").returncode, 0)
        proc = self.run_python(sync, "--write")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(outside.read_text(), "keep this file")
        for name in RUNNERS:
            dest = copied / name / "scripts/_utils.py"
            self.assertFalse(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), source.read_bytes())
        self.assertFalse((copied / "kermt-setup/scripts/_utils.py").exists())
        self.assertFalse((copied / "kermt-monitor/scripts/_utils.py").exists())
        self.assertEqual(self.run_python(sync, "--check").returncode, 0)


if __name__ == "__main__":
    unittest.main()
