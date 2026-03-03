# TextbookReasoning

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/EnvCommons/textbookreasoning) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/MegaScience/TextbookReasoning)

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

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Tasks

There is one split in this environment:

- **train**: 651,840 tasks

Questions span 7 STEM subjects with varying difficulty levels.

## Reward Structure

This is a single-turn environment. The agent submits an answer via the `submit_answer` tool. An LLM grader (gpt-5-mini) evaluates semantic correctness, handling equivalent formulations and LaTeX notation. Reward is binary: 1.0 if correct, 0.0 if incorrect.

## Data

Data consists of a Parquet file (~532 MB) sourced from [HuggingFace MegaScience/TextbookReasoning](https://huggingface.co/datasets/MegaScience/TextbookReasoning). Each row contains a question, subject category, and reference answer. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `submit_answer` | Submit your answer (may include LaTeX notation). Ends the episode. |

## Time Horizon

Single-turn. The agent reads the academic question and submits one answer.

## Environment Difficulty

TextbookReasoning evaluates graduate-level STEM reasoning across 7 academic subjects with semantic grading.

## Other Environment Requirements

OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.

## Safety

Agents in TextbookReasoning answer academic questions in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@misc{textbookreasoning2024,
  title={TextbookReasoning: A Large-Scale Academic Reasoning Dataset},
  author={MegaScience},
  year={2024},
  howpublished={https://huggingface.co/datasets/MegaScience/TextbookReasoning}
}
```
