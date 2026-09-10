## Description: <br>
Bootstrap the KERMT agent environment — verify host docker + nvidia-container-toolkit, build the kermt:latest image from the repo’s Dockerfile if it doesn’t yet exist, and run a GPU smoke test inside the container. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers setting up the KERMT molecular property prediction environment, verifying docker and GPU prerequisites, and building the container image before running training, finetuning, or inference workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [KERMT paper](https://arxiv.org/abs/2510.12719) <br>
- [GROVER paper](https://arxiv.org/abs/2007.02835) <br>
- [cuik-molmaker](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>
- [GROVER original implementation](https://github.com/tencent-ailab/grover) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) with 3 attempts each, run in isolated k8s-sandbox pods. Dataset: skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability, not Efficiency). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 84.9% | 87.8% |
| Security | 100.0% → 75.0% (-25.0 pp) | 62.5% → 100.0% (+37.5 pp) |
| Correctness | 40.0% → 100.0% (+60.0 pp) | 55.0% → 100.0% (+45.0 pp) |
| Discoverability | 91.7% | 91.7% |
| Effectiveness | 30.6% → 74.4% (+43.8 pp) | 39.4% → 67.5% (+28.1 pp) |
| Efficiency | 83.3% | 79.8% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
