## Description: <br>
Check progress for a detached KERMT run (pretrain, finetune, or any kermt_run_detached invocation). Reads run.json, queries docker for container state, tails the pretrain/finetune log, and parses progress lines (epoch, step, val loss). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers monitoring detached KERMT training and finetuning runs to check container status, training progress (epoch, step, val loss), and final test metrics. <br>

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
- [KERMT: Multitask finetuning and acceleration of chemical pretrained models for small molecule drug property prediction](https://arxiv.org/abs/2510.12719) <br>
- [GROVER: Self-Supervised Message Passing Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; optional structured JSON via --json flag] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Supports --json flag for machine-readable structured status output] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (3 positive, 1 negative), each attempted 3 times in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% behavior check). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.8% | 74.7% |
| Security | 100.0% → 100.0% (±0.0 points) | 50.0% → 75.0% (+25.0 points) |
| Correctness | 25.0% → 100.0% (+75.0 points) | 38.0% → 70.0% (+32.0 points) |
| Discoverability | 93.3% | 93.3% |
| Effectiveness | 42.5% → 68.1% (+25.6 points) | 24.0% → 41.3% (+17.3 points) |
| Efficiency | 97.7% | 93.8% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
