## Description: <br>
Continue KERMT pretraining on a custom SMILES corpus with a grover_base, cmim, or hybrid checkpoint. Use a local checkpoint or optionally download a pinned Hugging Face model bundle using HF_TOKEN if configured. Run containerized training and write model bundles, prepared data, logs, and checkpoints to user-selected host directories. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to continue pretraining KERMT molecular property prediction models on custom SMILES corpora using NVIDIA GPU-accelerated containerized workflows. <br>

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
- [Released Models](references/released-models.md) <br>
- [KERMT: Multitask finetuning and acceleration of chemical pretrained models](https://arxiv.org/abs/2510.12719) <br>
- [GROVER: Self-Supervised Message Passing Transformer](https://arxiv.org/abs/2007.02835) <br>
- [NV-KERMT-70M-v2 on Hugging Face](https://huggingface.co/nvidia/NV-KERMT-70M-v2) <br>
- [cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>
- [GROVER (original implementation)](https://github.com/tencent-ailab/grover) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Log file paths] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative) run in isolated sandbox pods with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.7% | 78.7% |
| Security | 88.5% → 100.0% (+11.5 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 18.5% → 84.0% (+65.5 points) | 58.0% → 92.0% (+34.0 points) |
| Discoverability | 90.0% | 77.5% |
| Effectiveness | 18.1% → 49.0% (+30.9 points) | 23.3% → 39.5% (+16.2 points) |
| Efficiency | 80.7% | 84.5% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
