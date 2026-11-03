# TextbookReasoning

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/TextbookReasoning) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/MegaScience/TextbookReasoning)

## Description

TextbookReasoning is an environment for evaluating graduate-level academic reasoning capabilities. It contains 651,840 questions across 7 STEM subjects: Medicine, Mathematics, Biology, Computer Science, Physics, Chemistry, and Economics. The environment supports LaTeX notation and uses LLM-based semantic grading.

## Capabilities

- Graduate-level academic reasoning
- Multi-subject STEM evaluation
- LaTeX notation support
- Semantic answer equivalence

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT). The underlying MegaScience/TextbookReasoning dataset is released under CC BY-NC-SA 4.0.

## Tasks

There are 651,840 tasks in a single `train` split. Questions span 7 STEM subjects.

## Reward Structure

This is a single-turn environment. The agent submits an answer via the `submit_answer` tool. An LLM grader (gpt-5-mini) evaluates semantic correctness, handling equivalent formulations and LaTeX notation. Reward is binary: 1.0 if correct, 0.0 if incorrect. Only the first graded answer is rewarded; a repeat submission gets a reward of -0.1. If the grader returns no verdict after 3 attempts, the answer is returned ungraded and the episode stays open so the agent can resubmit.

## Data

Data consists of a Parquet file (~532 MB) sourced from [HuggingFace MegaScience/TextbookReasoning](https://huggingface.co/datasets/MegaScience/TextbookReasoning). Each row contains a question, subject category, and reference answer. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `submit_answer` | Submit your answer (may include LaTeX notation). Ends the episode once the answer is graded. |

## Time Horizon

Single-turn. The agent reads the academic question and submits one answer.

## Other Environment Requirements

An OpenAI API key is required for LLM-based grading, passed as the `openai_api_key` secret.

## Safety

Agents in TextbookReasoning answer academic questions in a standard environment. The environment does not present direct safety risks.

## Citations

```bibtex
@article{fan2025megascience,
  title={MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning},
  author={Fan, Run-Ze and Wang, Zengzhi and Liu, Pengfei},
  journal={arXiv preprint arXiv:2507.16812},
  year={2025}
}
```
