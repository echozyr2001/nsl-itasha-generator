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
        
        # Prepare input for the predictor
        # We want the predictor to generate the INSTRUCTIONS for the image generator
        # based on the analysis of the images and the layout requirements.
        
        # Create a summary of the analysis to guide the prompt generation
        layout_desc = []
        layout_desc.append(f"Front/Back Divider: y={analysis.front_back_divider_y}%")
        for slot in analysis.layout_slots:
            layout_desc.append(
                f"Slot '{slot.slot_name}': Source Image {slot.source_images}, "
                f"Pos x={slot.position.x}%, y={slot.position.y}%, "
                f"Content: {slot.description}"
            )
        
        analysis_summary = (
            f"Synthesis Plan: {analysis.synthesis}\n"
            f"Layout Structure:\n" + "\n".join(layout_desc)
        )
        
        # Use the predictor to generate the prompt text
        # GEPA will optimize the signature/instructions of this predictor
        # to produce better prompts that yield higher scores.
        pred = self.compose_prompt(
            analysis_summary=analysis_summary,
            num_references=str(len(reference_paths)),
            num_slots=str(len(analysis.layout_slots))
        )
        
        return {"prompt": pred.prompt}
