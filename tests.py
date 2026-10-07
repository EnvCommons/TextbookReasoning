"""
Grader-handling tests for TextbookReasoning. The dataset row and the OpenAI
client are replaced with fakes, so these run without the parquet or an API key.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import textbookreasoning as mod
from textbookreasoning import SubmitAnswerInput, TextbookReasoning

REFERENCE = "42 joules"


class FakeCompletions:
    """Stands in for client.chat.completions; returns the scripted replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def create(self, **kwargs):
        if not self.replies:
            raise AssertionError("grader called more times than scripted")
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _env(monkeypatch, replies: list[str]) -> tuple[TextbookReasoning, FakeCompletions]:
    monkeypatch.setattr(mod, "get_data_path", lambda: Path("unused.parquet"))
    monkeypatch.setattr(mod, "read_one_row", lambda path, idx, cols: {
        "question": "How much work is done?",
        "reference_answer": REFERENCE,
        "answer": "Detailed solution ending in " + REFERENCE,
        "subject": "physics",
    })
    env = TextbookReasoning(task_spec={"id": "textbook_0"}, secrets={"openai_api_key": "test"})
    completions = FakeCompletions(replies)
    env.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return env, completions


def _submit(env: TextbookReasoning, answer: str):
    return asyncio.run(env.submit_answer(SubmitAnswerInput(answer=answer)))


def test_correct_verdict_scores_one(monkeypatch):
    env, client = _env(monkeypatch, ["Matches. <answer>CORRECT</answer>"])
    out = _submit(env, "42 J")
    assert out.reward == 1.0 and out.finished is True and client.calls == 1


def test_incorrect_verdict_scores_zero(monkeypatch):
    env, client = _env(monkeypatch, ["Wrong value. <answer>INCORRECT</answer>"])
    out = _submit(env, "17 J")
    assert out.reward == 0.0 and out.finished is True and client.calls == 1


def test_verdictless_reply_is_resampled(monkeypatch):
    env, client = _env(monkeypatch, ["The answer matches the reference.", "Matches. <answer>CORRECT</answer>"])
    out = _submit(env, "42 J")
    assert out.reward == 1.0 and out.finished is True and client.calls == 2


def test_verdictless_on_every_attempt_is_not_graded(monkeypatch):
    env, client = _env(monkeypatch, ["", "no tags", "<answer>maybe</answer>"])
    out = _submit(env, "42 J")
    assert client.calls == mod.GRADER_ATTEMPTS
    assert out.finished is False and out.reward == 0.0
    assert "Not graded" in out.blocks[0].text
    assert REFERENCE not in out.blocks[0].text + str(out.metadata)


def test_ungraded_attempt_can_be_resubmitted(monkeypatch):
    env, client = _env(monkeypatch, [""] * mod.GRADER_ATTEMPTS + ["Wrong. <answer>INCORRECT</answer>"])
    assert _submit(env, "17 J").finished is False
    out = _submit(env, "17 J")
    assert out.reward == 0.0 and out.finished is True
    assert "already_submitted" not in out.metadata


def test_second_graded_submission_is_penalised(monkeypatch):
    env, client = _env(monkeypatch, ["<answer>CORRECT</answer>"])
    assert _submit(env, "42 J").reward == 1.0
    out = _submit(env, "42 J")
    assert out.reward == mod.REPEAT_SUBMISSION_PENALTY and client.calls == 1
