import os
import json
from typing import List, Union
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class ImageAnalysis(BaseModel):
    description: str = Field(description="Detailed visual description of the image")
    elements: List[str] = Field(description="List of key visual elements identified")
    style: str = Field(description="Art style of the image")
    colors: List[str] = Field(description="Dominant color palette (hex codes or names)")
    mood: str = Field(description="Overall mood or atmosphere")

class AnalysisResult(BaseModel):
    images: List[ImageAnalysis] = Field(description="Analysis for each input image, in the same order as provided")
    synthesis: str = Field(description="A synthesized concept combining elements from all images suitable for a Nintendo Switch Lite skin design. Suggest how to merge these styles/elements.")

class VisionService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=self.api_key)
        # User requested gemini-2.5-flash
        self.model = "gemini-2.5-flash" 

    def analyze_image(self, image_paths: Union[str, List[str]]) -> AnalysisResult:
        """
        Analyzes the input image(s) and returns a structured JSON response.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        
        if not images:
            raise ValueError("No valid images loaded")

        prompt = """
        You are an expert visual designer for "Itasha" (decorated vehicle/device) art.
        Analyze the provided image(s) to create a design for a Nintendo Switch Lite skin.
        
        For each image, provide:
        1. A detailed visual description.
        2. Key elements (characters, objects).
        3. Art style.
        4. Color palette.
        5. Mood.

        Then, provide a 'synthesis' that suggests how to combine these elements into a single cohesive design concept for the Switch Lite.
        The Switch Lite has a roughly 2:1 aspect ratio.
        """
        
        contents = [prompt]
        contents.extend(images)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisResult
                )
            )
            
            # Parse the JSON response into our Pydantic model
            return AnalysisResult.model_validate_json(response.text)

        except Exception as e:
            print(f"Warning: Vision model {self.model} failed ({e}). Trying fallback...")
            # Fallback without schema if necessary, or just re-raise for now as we want structured data.
            # For simplicity in this demo, let's try one more time with a looser instruction if strict schema fails?
            # Or just return raw text wrapped in a basic structure if it fails totally?
            # Let's assume it works or raise.
            raise e
