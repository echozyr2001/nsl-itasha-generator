from __future__ import annotations

import json
from pathlib import Path
from typing import List

import dspy

from src.services.generation import GenerationService
from src.services.vision import AnalysisResult

class PromptComposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = GenerationService(enable_client=False)

    def forward(self, analysis_json: str, reference_paths: List[str]):
        analysis = AnalysisResult.model_validate_json(analysis_json)
        abs_refs = [
            str(Path(ref if Path(ref).is_absolute() else Path('assets') / ref).resolve())
            for ref in reference_paths
        ]
        prompt_parts = self.generator._build_generation_parts(analysis, abs_refs)
        text_parts = []
        for part in prompt_parts:
            text = getattr(part, 'text', None)
            if text:
                text_parts.append(text)
        prompt = "\n".join(text_parts)
        return {"prompt": prompt}
