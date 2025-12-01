import os
import json
from typing import List, Union
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google.oauth2 import service_account

load_dotenv()

class ImageAnalysis(BaseModel):
    description: str = Field(description="Detailed visual description of the image")
    elements: List[str] = Field(description="List of key visual elements identified")
    style: str = Field(description="Art style of the image")
    colors: List[str] = Field(description="Dominant color palette (hex codes or names)")
    mood: str = Field(description="Overall mood or atmosphere")


class PositionRange(BaseModel):
    x: List[float] = Field(description="Two-value list [min%, max%] for horizontal placement (0-100 scale)")
    y: List[float] = Field(description="Two-value list [min%, max%] for vertical placement (0-100 scale)")


class LayoutSlotDescription(BaseModel):
    slot_name: str = Field(description="Readable identifier for the slot")
    source_images: List[int] = Field(description="Indices (1-based) of reference images driving this slot")
    description: str = Field(description="What goes here (subjects, props, motifs)")
    purpose: str = Field(description="Why this slot exists / compositional intent")
    position: PositionRange
    avoid: str = Field(description="Mask regions to avoid within this slot (grey zones, central area)")


class AnalysisResult(BaseModel):
    images: List[ImageAnalysis] = Field(description="Analysis for each input image, in the same order as provided")
    synthesis: str = Field(description="A composition strategy. Do NOT suggest creating new styles. Explain how to arrange the EXISTING elements from the reference images into the mask's white zones.")
    layout_slots: List[LayoutSlotDescription] = Field(default_factory=list, description="Precise placement plan referencing the mask using percentage coordinates.")
    front_back_divider_y: float = Field(default=50.0, description="Y coordinate (0-100) where the top and bottom sections divide. This should be extracted from the mask's visible boundary line. The top section occupies y=0 to this value, bottom section occupies this value to y=100.")

class VisionService:
    def __init__(self):
        # Load credentials from account.json
        service_account_file = "account.json"
        if not os.path.exists(service_account_file):
             # Fallback to env var if file not found, though user said they use account.json
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(f"Service account file '{service_account_file}' not found and GOOGLE_API_KEY not set.")
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Use vertexai and service account
            # The google-genai SDK supports vertexai backend.
            # We need to configure the client for Vertex AI.
            # Typically this involves project and location, and credentials.
            
            # Read project_id from account.json
            with open(service_account_file, 'r') as f:
                info = json.load(f)
                self.project_id = info.get("project_id")

            self.location = "global" # Changed from us-central1 per user request
            
            # Define scopes explicitly
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file, 
                scopes=scopes
            )
            
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )


        # User requested gemini-2.5-flash
        self.model = "gemini-2.5-flash" 

    def analyze_image(self, image_paths: Union[str, List[str]], mask_path: str = None) -> AnalysisResult:
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

        mask_image = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask_image = Image.open(mask_path)
            except Exception as mask_err:
                print(f"Warning: failed to load mask {mask_path}: {mask_err}")
        elif not mask_path:
             # Fallback to default if not provided
             default_mask = os.path.join("assets", "cover.png")
             if os.path.exists(default_mask):
                 try:
                     mask_image = Image.open(default_mask)
                 except: pass

        prompt = """
        You are an expert layout planner for custom rectangular skins.
        After this text you will receive (optionally) the printable mask (white = printable, grey = cut out), 
        followed by N reference images in the exact order supplied by the user.

        TASK:
        1. Analyze the Reference Images to identify the MAIN SUBJECTS (characters, faces).
        2. Analyze the Mask to identify the WHITE ZONES (Safe Areas).
        3. Create a Layout Plan that maps each Subject to a specific White Zone.

        LAYOUT RULES:
        - The Mask has a Top Half (Front) and a Bottom Half (Back).
        - Top Half: Usually has two side zones (Left Grip, Right Grip). Place primary characters here.
        - Bottom Half: Usually one large zone. Place secondary characters or large emblems here.
        - AVOID: Do not place faces in the Grey Zones or the center of the Top Half (often a screen cutout).

        OUTPUT JSON:
        {
          "images": [
            {
              "description": "Brief description of the subject.",
              "elements": ["Key features to preserve"],
              "style": "Art style",
              "colors": ["Dominant colors"],
              "mood": "Mood"
            }
          ],
          "synthesis": "Plan: Place Subject 1 on Front-Left, Subject 2 on Front-Right...",
          "layout_slots": [
            {
              "slot_name": "Front-Left Zone",
              "source_images": [1],
              "description": "The main character from Image 1",
              "purpose": "Primary focal point",
              "position": { "x": [0, 45], "y": [0, 52] },
              "avoid": "Center screen area"
            }
          ],
          "front_back_divider_y": 52.5
        }
        
        CRITICAL: Analyze the mask image carefully. Identify the horizontal divider line that separates the top section from the bottom section. This divider is typically visible as a clear boundary or gap in the mask. Measure its Y coordinate as a percentage (0-100) and set "front_back_divider_y" to this exact value.
        """

        contents = [prompt]
        if mask_image:
            contents.append(mask_image)
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
            raise e
