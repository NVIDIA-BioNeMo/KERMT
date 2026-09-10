# Skill Benchmark: kermt-embed

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `kermt-embed`
- Evaluation date: 2026-09-10
- Evaluator version: `1.5.5`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:150b82ffdc6a29ff51ba6509f4ed70d6074e5944b957cdbd6c76f42ea048a482` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.6% — baseline ran, but no comparable score was available; uplift unavailable | 70.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 95.8% → 100.0% (+4.2 points) | 81.8% → 66.7% (-15.1 points) |
| Correctness | 18.3% → 80.0% (+61.7 points) | 60.0% → 73.3% (+13.3 points) |
| Discoverability | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 18.8% → 50.0% (+31.2 points) | 25.7% → 44.6% (+18.9 points) |
| Efficiency | 80.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,938,868 | 3,093,293 | N/A | N/A | skill 5/5; base 12/12 |
| claude-code | kermt-embed-001 | 394,186 | 455,578 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-002 | 352,190 | 342,459 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-003 | 365,070 | 554,161 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-004 | 447,119 | 317,222 | +129,897 | +40.95% | skill 1/1; base 1/1 |
| claude-code | kermt-embed-005 | 380,303 | 1,423,873 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 977,420 | 8,354,319 | N/A | N/A | skill 6/6; base 11/11 |
| codex | kermt-embed-001 | 125,655 | 343,876 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-embed-002 | 29,999 | 104,256 | -74,257 | -71.23% | skill 1/1; base 1/1 |
| codex | kermt-embed-003 | 116,213 | 390,151 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-embed-004 | 301,438 | 53,740 | +247,698 | +460.92% | skill 1/1; base 1/1 |
| codex | kermt-embed-005 | 404,115 | 7,462,296 | N/A | N/A | skill 2/2; base 3/3 |
| ALL AGENTS | Dataset aggregate | 2,916,288 | 11,447,612 | N/A | N/A | skill 11/11; base 23/23 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/kermt-embed/SKILL.md`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
