# Agent integrations

Agent-driven workflows for KERMT live under [`skills/`](skills/README.md). The
skills follow the [agentskills.io](https://agentskills.io) spec — one
directory per skill at `skills/<skill-name>/SKILL.md`.

Available workflows:
- `kermt-setup` — bootstrap the kermt container (run first)
- `kermt-continue-pretrain` — continue pretraining from an existing KERMT ckpt
- `kermt-pretrain-scratch` — pretrain from scratch on a custom corpus
- `kermt-add-cmim-pretrain` — add a cMIM decoder to a grover_base ckpt
- `kermt-finetune` — finetune a pretrained ckpt on a labeled CSV
- `kermt-infer` — predictions with a finetuned ckpt
- `kermt-embed` — extract molecular embeddings from any encoder-bearing ckpt
- `kermt-monitor` — tail logs / report progress for a detached run

See [`skills/README.md`](skills/README.md) for the full guide: hardware
requirements per workflow, how to install the skills with Claude Code /
Codex / Nemotron, and how to invoke the underlying scripts directly without
an agent.

Shared helpers and defaults are maintained in `skills/_shared/`. After editing
them, run `python3 skills/_shared/sync_shared.py --write` and commit the
per-skill copies; `--check` verifies they match. Each skill must use its
own bundled files so it can be installed and signed independently.
Development tests remain under `agent/tests/`.
