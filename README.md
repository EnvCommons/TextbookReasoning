# TextbookReasoning Environment

OpenReward environment for the TextbookReasoning dataset - a large-scale academic reasoning benchmark with 651,840 questions across 7 STEM subjects.

## Overview

TextbookReasoning evaluates agent reasoning capabilities on graduate-level academic questions spanning:
- Medicine
- Mathematics
- Biology
- Computer Science
- Physics
- Chemistry
- Economics

The environment uses **LLM-based grading** (gpt-5-mini) for semantic evaluation, allowing for flexible answer formats including LaTeX notation.

## Features

- **Large-scale evaluation**: 651,840 examples from train split
- **Multi-subject coverage**: 7 STEM disciplines
- **Semantic grading**: LLM judges answer correctness
- **LaTeX support**: Handles mathematical notation like `\boxed{}`
- **Single-turn Q&A**: Simple question → answer format

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/EnvCommons/textbookreasoning.git
cd textbookreasoning

# Install dependencies
pip install -r requirements.txt

# Download data (see DATA_UPLOAD.md for instructions)
python download_data.py

# Start server
python server.py
```

### Docker

```bash
# Build image
docker build -t textbookreasoning:latest .

# Run container (requires data mount)
docker run -p 8080:8080 \
  -v /path/to/orwd_data:/orwd_data \
  textbookreasoning:latest
```

## Usage

### Testing Locally

```bash
export OPENAI_API_KEY="sk-..."
python test_agent.py
```

### Using with OpenReward

```python
from openreward import OpenReward
import asyncio

async def test():
    client = OpenReward()

    env = client.environments.get(name="EnvCommons/textbookreasoning")
    tasks = await env.list_tasks(split="train")

    # Test first task
    async with env.session(
        task=tasks[0],
        secrets={"openai_api_key": "sk-..."}
    ) as session:
        prompt = await session.get_prompt()
        print(prompt[0].text)

        # Submit answer
        result = await session.call_tool(
            "submit_answer",
            {"answer": "Your answer here"}
        )
        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.blocks[0].text}")

asyncio.run(test())
```

## Data Requirements

The environment requires a 532 MB parquet file with the TextbookReasoning dataset. See [DATA_UPLOAD.md](DATA_UPLOAD.md) for detailed instructions.

**Summary:**
- Download from HuggingFace
- Upload to `/orwd_data/textbookreasoning/textbookreasoning_train.parquet`
- 651,840 examples from train split

## Environment Details

### Task Structure

Each task contains:
- `id`: Unique identifier (e.g., `textbook_math_12345`)
- `question`: The problem statement
- `subject`: Subject category

### Tool: submit_answer

**Input:**
- `answer` (string): Your final answer, may include LaTeX notation

**Output:**
- `reward`: 1.0 if correct, 0.0 if incorrect
- `blocks`: Grader reasoning + result
- `metadata`: Full grading details including reference answer

### Grading System

Answers are evaluated by gpt-5-mini using semantic comparison:
- Accepts equivalent formulations
- Handles LaTeX math notation
- Focuses on correctness, not exact wording
- Binary scoring: 1.0 (correct) or 0.0 (incorrect)

## API Reference

### Environment Class

```python
class TextbookReasoning(Environment):
    @classmethod
    def list_splits(cls) -> list[str]:
        """Returns: ["train"]"""

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Returns: List of 651,840 task specifications"""

    async def get_prompt(self) -> list[TextBlock]:
        """Returns: Question prompt"""

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """Submit answer for LLM grading"""
```

## Performance Considerations

- **Dataset size**: 532 MB loaded into memory at startup
- **Grading latency**: ~1-2 seconds per LLM grading call
- **Memory usage**: ~1-2 GB recommended for deployment
- **API costs**: Each submission uses gpt-5-mini (~$0.001 per grading)

## Development

### Running Tests

```bash
pytest golden_tests.py -v
```

### Code Structure

```
textbookreasoning/
├── textbookreasoning.py    # Main environment class
├── server.py               # Server wrapper
├── test_agent.py           # Test script
├── download_data.py        # Data download helper
├── requirements.txt        # Dependencies
├── Dockerfile              # Container definition
├── DATA_UPLOAD.md          # Data setup guide
└── README.md               # This file
```

## License

See dataset license on HuggingFace for usage terms: https://huggingface.co/datasets/MegaScience/TextbookReasoning

The dataset is licensed under CC-BY-NC-SA-4.0 (non-commercial use).

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Support

For issues:
- Environment bugs: GitHub issues
- Dataset questions: HuggingFace dataset page
- OpenReward platform: https://docs.openreward.org/

## Citation

If you use this environment, please cite the original TextbookReasoning dataset:

```bibtex
@misc{textbookreasoning2024,
  title={TextbookReasoning: A Large-Scale Academic Reasoning Dataset},
  author={MegaScience},
  year={2024},
  howpublished={https://huggingface.co/datasets/MegaScience/TextbookReasoning}
}
```
