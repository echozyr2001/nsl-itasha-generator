from __future__ import annotations

import json
from pathlib import Path

import dspy
from dspy.optimizers import GEPA

from src.prompt_optim.prompt_composer import PromptComposer
from src.prompt_optim.eval_rules import score_prompt

DATASET_PATH = Path("datasets/gepa_dataset.json")


def load_dataset():
    return json.loads(DATASET_PATH.read_text())


def eval_fn(program, example):
    analysis_json = json.dumps(example["analysis"])
    result = program(analysis_json=analysis_json, reference_paths=example["references"])
    prompt = result["prompt"]
    return score_prompt(prompt)


def main():
    trainset = load_dataset()
    program = PromptComposer()
    optimizer = GEPA(max_mutations=3, num_rounds=3)
    optimized = optimizer.compile(program, trainset=trainset, eval_fn=lambda prog, ex: eval_fn(prog, ex))
    Path("datasets/gepa_optimized_prompt.txt").write_text(optimized(analysis_json=json.dumps(trainset[0]["analysis"]), reference_paths=trainset[0]["references"])["prompt"])

if __name__ == "__main__":
    main()
