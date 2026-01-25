"""
TextbookReasoning Environment

Large-scale academic reasoning benchmark spanning 7 STEM subjects with detailed answers.
Uses LLM-based grading for semantic evaluation of student responses.

Dataset: HuggingFace TextbookReasoning (train split, 651k examples)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

import pyarrow.parquet as pq

def read_one_row(path: Path, idx: int, columns: list[str]):
    pf = pq.ParquetFile(path)  # reads metadata/footer, not full data
    remaining = idx
    rg = 0
    while rg < pf.num_row_groups:
        n = pf.metadata.row_group(rg).num_rows
        if remaining < n:
            break
        remaining -= n
        rg += 1

    table = pf.read_row_group(rg, columns=columns)   # loads only this row-group + these cols
    # slice to 1 row, convert minimal amount to python/pandas
    row = table.slice(remaining, 1).to_pydict()
    return {k: row[k][0] for k in row}

# Path configuration: support both local dev and production
if os.path.exists("/orwd_data/"):
    PATH = Path("/orwd_data/")
else:
    PATH = Path(__file__).parent


# Grader prompt template
GRADER_TEMPLATE = """You are an expert academic grader evaluating student answers across multiple STEM subjects.

Your task: Determine if the student's answer is semantically equivalent to the reference answer.

**Question:**
{question}

**Reference Answer:**
{reference_answer}

**Student Answer:**
{student_answer}

**Grading Instructions:**
1. Focus on semantic correctness, not exact wording
2. For math: Accept equivalent expressions (e.g., "1/2" = "0.5")
3. For LaTeX: \\boxed{{x}} answers should match the boxed content
4. For scientific concepts: Accept paraphrased but accurate explanations
5. Ignore formatting differences (capitalization, whitespace, etc.)

**Output Format:**
First, provide a brief analysis (2-3 sentences) of your reasoning.
Then, on a new line, write EXACTLY one of:
- "CORRECT" if the student answer is semantically correct
- "INCORRECT" if the student answer is wrong or incomplete

Analysis:
"""


def get_data_path() -> Path:
    """
    Get path to the parquet data file.

    Returns:
        Path to the parquet file

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    data_path = PATH / "data" / "textbookreasoning_train.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please upload textbookreasoning_train.parquet to /orwd_data/textbookreasoning/\n"
            f"See DATA_UPLOAD.md for instructions."
        )

    return data_path


class TaskSpec(BaseModel):
    """Task specification schema."""
    id: str


class SubmitAnswerInput(BaseModel):
    """Input schema for submit_answer tool."""
    answer: str = Field(
        ...,
        description="Your final answer to the question. Can include LaTeX notation if needed."
    )


class TextbookReasoning(Environment):
    """
    TextbookReasoning environment for academic STEM questions.

    Covers 7 subjects: Medicine, Math, Physics, Chemistry, Biology,
    Engineering, and Computer Science.

    Uses LLM-based grading for semantic evaluation.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        # Extract task index from ID (format: textbook_{idx})
        task_idx = int(self.validated.id.split('_')[1])

        # Load only this specific row using efficient row-group reading
        cols = ["question", "reference_answer", "answer", "subject"]
        task_row = read_one_row(get_data_path(), task_idx, cols)

        self.question = str(task_row["question"])
        self.reference_answer = str(task_row.get("reference_answer", ""))
        self.detailed_answer = str(task_row.get("answer", ""))
        self.subject = str(task_row["subject"])

        # Validate OpenAI API key
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for LLM grading. "
                "Pass via secrets parameter: secrets={'openai_api_key': 'sk-...'}"
            )

        self.client = openai.AsyncClient(api_key=api_key)

    async def _grade_sample(self, student_answer: str) -> dict:
        """
        Grade student answer using LLM grader.

        Args:
            student_answer: The student's submitted answer

        Returns:
            dict with keys:
                - is_correct: bool
                - grading_response: str (full LLM reasoning)
                - reference_answer: str (for metadata)
        """
        # Get reference answer for this task
        reference = self.reference_answer

        # Handle empty reference_answer: fall back to detailed answer
        if not reference or reference.strip() == "":
            reference = self.detailed_answer

        grader_prompt = GRADER_TEMPLATE.format(
            question=self.question,
            reference_answer=reference,
            student_answer=student_answer,
        )

        # Use gpt-5-mini for grading (fast and cost-effective)
        res = await self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": grader_prompt}
            ],
        )

        grading_response = res.choices[0].message.content or ""

        # Parse response: look for "CORRECT" without "INCORRECT"
        upper_response = grading_response.upper()
        is_correct = "CORRECT" in upper_response and "INCORRECT" not in upper_response

        return {
            "is_correct": is_correct,
            "grading_response": grading_response,
            "reference_answer": reference
        }

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final answer for grading.

        The answer will be evaluated by an LLM grader that checks for semantic
        correctness. You can use LaTeX notation (e.g., \\boxed{}) for math answers.
        """
        grader_output = await self._grade_sample(params.answer)

        # Binary reward
        reward = 1.0 if grader_output["is_correct"] else 0.0

        # Display grader reasoning + result
        result_text = "✅ Correct" if grader_output["is_correct"] else "❌ Incorrect"

        return ToolOutput(
            blocks=[
                TextBlock(
                    text=f"{grader_output['grading_response']}\n\n{result_text}"
                )
            ],
            metadata={
                "task_id": self.validated.id,
                "subject": self.subject,
                "student_answer": params.answer,
                "reference_answer": grader_output["reference_answer"],
                "is_correct": grader_output["is_correct"],
                "grader_reasoning": grader_output["grading_response"]
            },
            reward=reward,
            finished=True
        )

    async def get_prompt(self) -> list[TextBlock]:
        """Return the question prompt."""
        return [TextBlock(text=self.question)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        List all tasks for the given split.

        Uses only parquet metadata (row count) - no data loaded.
        """
        if split != "train":
            return []

        # Get row count from parquet metadata (fast, no data read)
        pf = pq.ParquetFile(get_data_path())
        total_rows = pf.metadata.num_rows

        # Generate task specs based on row indices only
        tasks = []
        for idx in range(total_rows):
            tasks.append({
                "id": f"textbook_{idx}"
            })

        return tasks

    @classmethod
    def list_splits(cls) -> list[str]:
        """List available splits."""
        return ["train"]
