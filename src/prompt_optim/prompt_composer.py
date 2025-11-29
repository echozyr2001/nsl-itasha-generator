from __future__ import annotations

import json
from pathlib import Path
from typing import List

import dspy
from dspy import InputField, OutputField

from src.services.generation import GenerationService
from src.services.vision import AnalysisResult


class PromptComposer(dspy.Module):
    """Composes generation prompts from analysis results and reference images."""
    
    def __init__(self):
        super().__init__()
        self.generator = GenerationService(enable_client=False)
        # Use a simple Predict (not ChainOfThought) for better GEPA compatibility
        # This predictor will be optimized by GEPA to improve prompt quality
        self.compose_prompt = dspy.Predict(
            "analysis_summary, num_references, num_slots -> prompt"
        )

    def forward(
        self, 
        analysis_json: str = InputField(desc="JSON string of AnalysisResult"),
        reference_paths: List[str] = InputField(desc="List of reference image paths")
    ) -> dict:
        """
        Generate a prompt from analysis results and reference images.
        
        Returns:
            dict with "prompt" key containing the full prompt text
        """
        try:
            analysis = AnalysisResult.model_validate_json(analysis_json)
        except Exception as e:
            raise ValueError(f"Failed to parse analysis_json: {e}")
        
        # Resolve reference paths (handle both absolute and relative)
        abs_refs = []
        for ref in reference_paths:
            ref_path = Path(ref)
            if ref_path.is_absolute():
                abs_refs.append(str(ref_path.resolve()))
            else:
                # Try relative to project root or assets directory
                candidate = Path('assets') / ref
                if candidate.exists():
                    abs_refs.append(str(candidate.resolve()))
                else:
                    abs_refs.append(str(Path(ref).resolve()))
        
        # Build base prompt parts using GenerationService
        prompt_parts = self.generator._build_generation_parts(analysis, abs_refs)
        
        # Extract text from parts to create base prompt
        text_parts = []
        for part in prompt_parts:
            text = getattr(part, 'text', None)
            if text:
                text_parts.append(text)
        
        base_prompt_text = "\n\n".join(text_parts)
        
        # Use the predictor (GEPA will optimize its instructions)
        # For now, we use the base prompt, but GEPA can optimize the predictor's behavior
        # The predictor is present so GEPA can identify and optimize it
        try:
            analysis_summary = f"Layout: {len(analysis.layout_slots)} slots, divider at y={analysis.front_back_divider_y}%"
            result = self.compose_prompt(
                analysis_summary=analysis_summary,
                num_references=str(len(abs_refs)),
                num_slots=str(len(analysis.layout_slots))
            )
            # The predictor output is not used for now - GEPA optimizes its instructions
            # but we return the base prompt built from GenerationService
        except Exception:
            pass
        
        prompt = base_prompt_text  # Use base prompt from GenerationService
        
        return {"prompt": prompt}
