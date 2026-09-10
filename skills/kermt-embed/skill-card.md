## Description: <br>
Extract per-molecule embeddings from any encoder-bearing KERMT checkpoint, running containerized extraction to produce per-readout .npy embeddings, canonical SMILES, and validity arrays. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational chemists who need to extract molecular embeddings from KERMT checkpoints for downstream property prediction, similarity search, or integration into cheminformatics pipelines. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Released KERMT Models](references/released-models.md) <br>
- [Multitask finetuning and acceleration of chemical pretrained models (arXiv)](https://arxiv.org/abs/2510.12719) <br>
- [Self-Supervised Graph Transformer on Large-Scale Molecular Data — GROVER (arXiv)](https://arxiv.org/abs/2007.02835) <br>
- [NV-KERMT-70M-v2 on Hugging Face](https://huggingface.co/nvidia/NV-KERMT-70M-v2) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands] <br>
**Output Format:** [NumPy arrays (.npy) and JSON manifests with Markdown-formatted agent guidance] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Per-readout embeddings (atom_from_atom, bond_from_atom, atom_from_bond, bond_from_bond), canonical SMILES array, validity array, and a replayable run manifest] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative) with 3 attempts each, run in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved and the expected workflow behavior was followed (equal-weight mean of goal completion and behavior check). <br>
- Efficiency: Tool-call productivity and token efficiency (50% each). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.6% | 70.6% |
| Security | 95.8% → 100.0% (+4.2 points) | 81.8% → 66.7% (-15.1 points) |
| Correctness | 18.3% → 80.0% (+61.7 points) | 60.0% → 73.3% (+13.3 points) |
| Discoverability | 92.5% | 85.0% |
| Effectiveness | 18.8% → 50.0% (+31.2 points) | 25.7% → 44.6% (+18.9 points) |
| Efficiency | 80.5% | 83.5% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
