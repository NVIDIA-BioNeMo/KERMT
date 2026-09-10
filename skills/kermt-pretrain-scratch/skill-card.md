## Description: <br>
Pretrain a fresh KERMT model from scratch on a user-provided corpus, building a new vocabulary, instantiating the model architecture from defaults, and launching distributed pretraining inside the kermt container. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational chemists who need to pretrain a new KERMT molecular property prediction model from scratch on a custom chemistry corpus, rather than continuing from a released checkpoint. <br>

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
- [KERMT paper — Multitask finetuning and acceleration of chemical pretrained models](https://arxiv.org/abs/2510.12719) <br>
- [GROVER paper — Self-Supervised Graph Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835) <br>
- [cuik-molmaker — GPU-accelerated molecular featurization](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>
- [Original GROVER implementation](https://github.com/tencent-ailab/grover) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces a run manifest (run.json), training logs, and TensorBoard event files in the run directory] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) executed with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct, measured by final-answer accuracy against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed, including skill selection and decoy avoidance. <br>
- Effectiveness: Whether the skill helped complete the task, combining goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, combining tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity measured against expected tool usage. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage relative to baseline. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.4% | 79.8% |
| Security | 100.0% → 100.0% (±0.0 points) | 57.1% → 100.0% (+42.9 points) |
| Correctness | 6.7% → 95.0% (+88.3 points) | 68.6% → 100.0% (+31.4 points) |
| Discoverability | 91.7% | 86.7% |
| Effectiveness | 19.5% → 47.5% (+28.0 points) | 36.1% → 40.6% (+4.5 points) |
| Efficiency | 82.8% | 71.8% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
