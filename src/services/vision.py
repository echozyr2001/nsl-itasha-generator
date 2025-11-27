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
    avoid: str = Field(description="Hardware regions to avoid within this slot (screen, buttons, vents)")


class AnalysisResult(BaseModel):
    images: List[ImageAnalysis] = Field(description="Analysis for each input image, in the same order as provided")
    synthesis: str = Field(description="A synthesized concept combining elements from all images suitable for a custom handheld console skin design. Suggest how to merge these styles/elements.")
    layout_slots: List[LayoutSlotDescription] = Field(default_factory=list, description="Precise placement plan referencing the device mask using percentage coordinates.")
    front_back_divider_y: float = Field(default=50.0, description="Y coordinate (0-100) where the front (top) and back (bottom) sections divide. This should be extracted from the mask's visible boundary line. The front section occupies y=0 to this value, back section occupies this value to y=100.")

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

        mask_path = os.path.join("assets", "cover.png")
        mask_image = None
        if os.path.exists(mask_path):
            try:
                mask_image = Image.open(mask_path)
            except Exception as mask_err:
                print(f"Warning: failed to load mask {mask_path}: {mask_err}")

        prompt = """
        You are an expert layout planner for custom handheld console skins.
        After this text you will receive (optionally) the printable device mask (white = printable, grey = cut out), 
        followed by N reference images in the exact order supplied by the user.

        Behave like a meticulous data labeler:
        - For each reference image, exhaustively list identifiable characters, outfits, props, logos, patterns, lighting, and stylistic motifs.
        - Capture distinguishing traits (hair, clothing colors, accessories, poses, expressions) so a synthesis model can faithfully recreate them.
        - Identify background textures, gradients, effects, and any typography.

        When combining references, plan a single cohesive palette and lighting treatment. Describe seam transitions and blending strategies for where controller halves meet and where front/back surfaces connect.
        Treat the printable mask as law: the goal is to concentrate focal characters entirely inside the white regions (both left/right grips on the front, full expanse on the back), while using the grey zones only for soft background carryover.
        The top half of the canvas is the FRONT face and must carry the primary composition stretching to the far left/right corners; the bottom half is the BACK panel and can host secondary motifs or emblems.
        
        CRITICAL FOR MULTIPLE IMAGES: If multiple reference images are provided, you must create a unified composition that:
        1. Extracts distinct elements from each image (characters, props, backgrounds, logos, text) and assigns them to specific layout slots
        2. Maintains visual consistency across all elements (same character should look identical across all instances)
        3. Creates a cohesive color palette that works across all extracted elements
        4. Distributes elements strategically: use layout_slots to place characters from different reference images in different positions (e.g., one character on front-left, another on front-right, a third on the back)
        5. If a reference image contains multiple characters or elements, you may use different elements from the same image in different slots
        6. The synthesis field should explicitly describe how elements from each reference image will be combined and where they will be placed

        You MUST return JSON strictly following this schema:
        {
          "images": [
            {
              "description": "...",
              "elements": ["..."],
              "style": "...",
              "colors": ["..."],
              "mood": "..."
            }
          ],
          "synthesis": "...",
          "layout_slots": [
            {
              "slot_name": "...",
              "source_images": [1,2],
              "description": "...",
              "purpose": "...",
              "position": { "x": [minPercent, maxPercent], "y": [minPercent, maxPercent] },
              "avoid": "Describe the mask / hardware areas that must remain empty for this slot"
            }
          ],
          "front_back_divider_y": 52.5
        }

        Requirements:
        - Percentages reference the full 1:1 canvas (0,0 top-left, 100,100 bottom-right). Always cross-check the mask and keep key subjects outside grey (cut-out) regions.
        - Express coordinates as ranges with at least 5% precision (e.g., x:[5,22], y:[60,85]). Avoid vague terms like "left" or "bottom" without numeric bounds.
        - layout_slots must describe both front (top half) and back (bottom half) placements with precise coords. Ensure every major character, logo, or emblem sits wholly inside a white mask region, and dedicate additional slots to fill remaining white space (especially the large front corners and the expansive back panel) with meaningful artwork.
        - Call out explicit slots (or background treatments) that cover the front-left and front-right corners so they never appear blank.
        - Mention color or lighting gradients needed to hide seams between slots.
        - Use the provided image order: slot.source_images should reference the 1-based index of the inspiration image.
        - CRITICAL: Analyze the mask image carefully. Identify the horizontal divider line that separates the front (top) section from the back (bottom) section. This divider is typically visible as a clear boundary or gap in the mask. Measure its Y coordinate as a percentage (0-100) and set "front_back_divider_y" to this exact value. The front section occupies y=0 to front_back_divider_y, and the back section occupies front_back_divider_y to y=100. This divider position is crucial for aligning the generated texture's composition split.
        - FOR MULTIPLE IMAGES: Create enough layout_slots to utilize elements from ALL reference images. Each major element (character, prop, logo) from each reference image should have at least one dedicated slot. The synthesis field must explicitly state: "From Image 1, extract [elements] and place them in [slots]. From Image 2, extract [elements] and place them in [slots]." This ensures the generation model knows exactly which elements come from which reference image.
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
