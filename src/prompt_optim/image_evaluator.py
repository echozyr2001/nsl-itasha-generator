"""Image generation and evaluation for GEPA optimization."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

from src.services.generation import GenerationService
from src.services.vision import AnalysisResult


class ImageEvaluator:
    """Evaluates prompts by generating images and scoring them."""
    
    def __init__(self):
        """Initialize the image evaluator with generation and scoring services."""
        self.generator = GenerationService(enable_client=True)
        
        # Initialize scoring client (gemini-3-pro-preview for text/image analysis)
        service_account_file = "account.json"
        if not os.path.exists(service_account_file):
            raise ValueError(f"Service account file '{service_account_file}' not found.")
        
        with open(service_account_file, 'r') as f:
            info = json.load(f)
            self.project_id = info.get("project_id")
        
        self.location = "global"
        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=scopes
        )
        
        self.scoring_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=credentials
        )
        self.scoring_model = "gemini-3-pro-preview"
    
    def _image_part_from_bytes(self, image_bytes: bytes, mime_type: str = "image/png"):
        """Convert image bytes to genai Part."""
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    def _generate_image_with_custom_prompt(
        self,
        analysis_result: AnalysisResult,
        reference_images: list[str],
        prompt_text: str,
        output_path: str
    ) -> str | None:
        """
        Generate an image using a custom prompt text while preserving image parts.
        
        Args:
            analysis_result: The analysis result
            reference_images: List of reference image paths
            prompt_text: Custom prompt text (optimized)
            output_path: Path to save the generated image
        
        Returns:
            Path to generated image or None on failure
        """
        # Parse prompt text into parts
        # The prompt_text contains all text instructions
        # We need to add image parts (mask, examples, reference images)
        
        # Split prompt text by double newlines to get logical sections
        text_sections = prompt_text.split("\n\n")
        
        # Build parts list
        parts = []
        
        # Add all text sections as parts
        for section in text_sections:
            section = section.strip()
            if section:
                parts.append(types.Part.from_text(text=section))
        
        # Add template mask if available
        mask_path = Path("assets/cover.png")
        if mask_path.exists():
            parts.append(types.Part.from_text(
                text="Template mask placement aid ONLY: white = survives after cutting, grey = will be removed. "
                "Use it solely to position characters and patterns. Do not copy its shapes, colors, or transparency into the final artwork."
            ))
            parts.append(self.generator._image_part_from_path(str(mask_path)))
        
        # Add example pairs
        for texture_path, preview_path, note in self.generator.example_pairs:
            if Path(texture_path).exists():
                parts.append(types.Part.from_text(
                    text=f"Example: '{texture_path}' is the TARGET printable vinyl texture (what you must create)."
                ))
                parts.append(self.generator._image_part_from_path(texture_path))
        
        # Add reference images
        for idx, img_path in enumerate(reference_images):
            if Path(img_path).exists():
                try:
                    parts.append(self.generator._image_part_from_path(img_path))
                except Exception as e:
                    print(f"Warning: Failed to load reference image {img_path}: {e}", file=sys.stderr)
        
        # Generate image
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.generator.client.models.generate_content(
                    model=self.generator.image_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        response_modalities=[types.Modality.IMAGE],
                        image_config=types.ImageConfig(aspect_ratio="1:1")
                    )
                )
                
                # Extract image from response
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            img_data = part.inline_data.data
                            img = Image.open(BytesIO(img_data))
                            img.save(output_path)
                            return output_path
                
                print("Warning: No inline image data returned from model", file=sys.stderr)
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
            except Exception as e:
                print(f"Error generating image with custom prompt (attempt {attempt}/{max_attempts}): {e}", file=sys.stderr)
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
                return None
        return None
    
    def generate_and_evaluate(
        self,
        analysis_result: AnalysisResult,
        reference_images: list[str],
        prompt_text: str,
        target_texture_path: str | None = None
    ) -> tuple[float, str]:
        """
        Generate an image using the prompt and evaluate it.
        
        Args:
            analysis_result: The analysis result with layout information
            reference_images: List of reference image paths
            prompt_text: The optimized prompt text to use
            target_texture_path: Optional path to target texture for comparison
        
        Returns:
            Tuple of (score: float, feedback: str)
        """
        # Create temp directory for generated images
        temp_dir = Path("assets/output/gepa_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate image using the custom prompt
        temp_output = temp_dir / f"generated_{os.getpid()}_{abs(hash(prompt_text)) % 100000}.png"
        
        try:
            # Generate image with custom prompt
            result_path = self._generate_image_with_custom_prompt(
                analysis_result=analysis_result,
                reference_images=reference_images,
                prompt_text=prompt_text,
                output_path=str(temp_output)
            )
            
            if not result_path or not os.path.exists(result_path):
                return 0.0, "Failed to generate image"
            
            # Load generated image
            with open(result_path, 'rb') as f:
                generated_image_bytes = f.read()
            
            # Evaluate the generated image
            score, feedback = self._score_image(
                generated_image_bytes=generated_image_bytes,
                analysis_result=analysis_result,
                reference_images=reference_images,
                target_texture_path=target_texture_path
            )
            
            return score, feedback
            
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 0.0, f"Error during generation/evaluation: {str(e)}"
    
    def _score_image(
        self,
        generated_image_bytes: bytes,
        analysis_result: AnalysisResult,
        reference_images: list[str],
        target_texture_path: str | None = None
    ) -> tuple[float, str]:
        """
        Score a generated image using gemini-3-pro-preview.
        
        Args:
            generated_image_bytes: The generated image as bytes
            analysis_result: The analysis result
            reference_images: List of reference image paths
            target_texture_path: Optional path to target texture for comparison
        
        Returns:
            Tuple of (score: float, feedback: str)
        """
        # Build scoring prompt
        scoring_prompt = """
    You are evaluating a generated texture image for a handheld console skin design.

Evaluation Criteria:
1. **Layout Compliance** (30%): Does the image follow the specified layout slots and avoid forbidden zones (screen, buttons)?
2. **Reference Image Usage** (25%): Are elements from the reference images correctly integrated and positioned?
3. **Visual Quality** (20%): Is the image visually appealing, with good color harmony and composition?
4. **Technical Requirements** (15%): Is it a clean 1:1 base texture without mask outlines, hardware depictions, or transparent areas?
5. **Composition Split** (10%): Is the front/back divider correctly aligned at the specified Y coordinate?

Provide a score from 0.0 to 1.0 and detailed feedback.

Return your response as JSON with the following structure:
{
  "score": 0.0-1.0,
  "feedback": "Detailed feedback explaining the score",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"]
}
"""
        
        # Add layout information
        scoring_prompt += f"\n\nLayout Plan:\n"
        scoring_prompt += f"Front/Back Divider: y={analysis_result.front_back_divider_y}%\n"
        for slot in analysis_result.layout_slots:
            scoring_prompt += f"- {slot.slot_name}: position x={slot.position.x}, y={slot.position.y}, purpose: {slot.purpose}, avoid: {slot.avoid}\n"
        
        # Build parts for scoring
        parts = [types.Part.from_text(text=scoring_prompt)]
        
        # Add generated image
        parts.append(self._image_part_from_bytes(generated_image_bytes))
        
        # Add target texture if provided for comparison
        if target_texture_path and os.path.exists(target_texture_path):
            with open(target_texture_path, 'rb') as f:
                target_bytes = f.read()
            parts.append(types.Part.from_text(text="\n\nTarget texture for comparison (this is what we're trying to achieve):"))
            parts.append(self._image_part_from_bytes(target_bytes))
        
        # Call scoring model
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.scoring_client.models.generate_content(
                    model=self.scoring_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                # Extract JSON response
                if response.candidates and response.candidates[0].content:
                    text = response.candidates[0].content.parts[0].text
                    result = json.loads(text)
                    
                    score = float(result.get("score", 0.0))
                    feedback = result.get("feedback", "No feedback provided")
                    
                    return score, feedback
                else:
                    print("Warning: No scoring response candidates", file=sys.stderr)
                    if attempt < max_attempts:
                        time.sleep(2 * attempt)
                        continue
            except Exception as e:
                print(f"Error in scoring attempt {attempt}/{max_attempts}: {e}", file=sys.stderr)
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                    continue
                return 0.0, f"Error in scoring: {str(e)}"
        return 0.0, "No response from scoring model"

