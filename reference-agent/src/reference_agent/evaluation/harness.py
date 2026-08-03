"""A real evaluation harness with a fixed task set.

The old book draft defined many "evaluator classes" but never ran them on a
fixed dataset. Here evaluation is a function over a fixed JSONL sample set,
producing a reproducible pass-rate report.
"""

from __future__ import annotations

import json
import os

from ..agent import Agent
from ..provider import FakeProvider

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


def load_samples(path: str | None = None) -> list[dict]:
    path = path or os.path.join(_DATA_DIR, "eval_samples.jsonl")
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_case(agent: Agent, sample: dict) -> dict:
    output = agent.run(sample["input"])
    needles = sample.get("expect_contains", [])
    passed = all(needle in output for needle in needles)
    return {"id": sample.get("id"), "passed": passed, "output": output}


def evaluate(agent: Agent | None = None, samples: list[dict] | None = None) -> dict:
    agent = agent or Agent(FakeProvider())
    samples = samples or load_samples()
    results = [run_case(agent, s) for s in samples]
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "results": results,
    }
