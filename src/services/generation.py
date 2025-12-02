import os
import json
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google.oauth2 import service_account
# We import AnalysisResult only for type hinting if needed, but to be safe with circulars (even if none now)
# we can just assume the object structure or use Any.
# For clarity, let's import it.
from src.services.vision import AnalysisResult

load_dotenv()

class GenerationService:
    def __init__(self, enable_client: bool = True):
        """
        Initialize GenerationService.
        
        Args:
            enable_client: If False, skip client initialization (useful for prompt optimization only).
        """
        self.client = None
        self.project_id = None
        self.location = None
        
        if enable_client:
            # Load credentials from account.json
            service_account_file = "account.json"
            if not os.path.exists(service_account_file):
                # Fallback to env var if file not found
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        f"Service account file '{service_account_file}' not found and GOOGLE_API_KEY not set."
                    )
                self.client = genai.Client(api_key=self.api_key)
                self.project_id = None
                self.location = None
            else:
                # Use vertexai and service account
                with open(service_account_file, 'r') as f:
                    info = json.load(f)
                    self.project_id = info.get("project_id")

                self.location = "global"  # Changed from us-central1 per user request

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

        # Path to the console template mask used to guide compositions
        self.template_mask_path = "assets/cover.png"
        if not os.path.exists(self.template_mask_path):
            self.template_mask_path = None

        # Example reference pairs (texture prior to masking, rendered preview after mask) to show the desired pipeline
        self.example_pairs = [
            ("assets/ref/2-b.JPG", "assets/ref/2-a.JPG", "Example pair demonstrating cheerful mascot layout; 2-b.JPG is the printable texture, 2-a.JPG shows how it appears after applying the mask."),
            ("assets/ref/8-b.JPG", "assets/ref/8-a.JPG", "Example pair showing chibi characters with soft background; 8-b.JPG is the printable texture, 8-a.JPG is the masked preview."),
            ("assets/ref/3-b.JPG", "assets/ref/3-a.JPG", "Example pair showing dynamic character placement."),
            ("assets/ref/4-b.JPG", "assets/ref/4-a.JPG", "Example pair showing balanced composition."),
        ]

        # User requested Gemini 3 Pro for the "Control" / Logic part
        self.logic_model = "gemini-3-pro-preview"
        
        # For final image synthesis we still call gemini-3-pro-image-preview
        self.image_model = "gemini-3-pro-image-preview"

    def _image_part_from_path(self, image_path: str):
        """
        Loads an image from disk and wraps it into a genai Part for upload.
        """
        img = Image.open(image_path)
        buffered = BytesIO()
        img.save(buffered, format=img.format or "PNG")
        img_bytes = buffered.getvalue()
        mime = f"image/{img.format.lower() if img.format else 'png'}"
        return types.Part.from_bytes(data=img_bytes, mime_type=mime)

    def _analysis_summary_text(self, analysis_result: AnalysisResult) -> str:
        lines = ["=== STRUCTURED ANALYSIS ==="]
        palette = []
        style_set = []
        mood_set = []
        for idx, img in enumerate(analysis_result.images):
            lines.append(f"\nImage {idx+1}:")
            lines.append(f"  Description: {img.description}")
            lines.append(f"  Elements: {', '.join(img.elements)}")
            lines.append(f"  Style: {img.style}")
            lines.append(f"  Colors: {', '.join(img.colors)}")
            lines.append(f"  Mood: {img.mood}")
            palette.extend([c for c in img.colors if c])
            if img.style and img.style not in style_set:
                style_set.append(img.style)
            if img.mood and img.mood not in mood_set:
                mood_set.append(img.mood)
        
        lines.append(f"\n=== SYNTHESIS PLAN ===")
        lines.append(f"{analysis_result.synthesis}")
        
        lines.append(f"\nLayout Structure:")
        lines.append(f"Front/Back Divider: y={analysis_result.front_back_divider_y}%")

        if palette:
            unique_palette = []
            for color in palette:
                if color not in unique_palette:
                    unique_palette.append(color)
            lines.append("Global Palette Seeds: " + ", ".join(unique_palette[:8]))
        if style_set or mood_set:
            cohesion_bits = []
            if style_set:
                cohesion_bits.append("Styles to blend: " + ", ".join(style_set))
            if mood_set:
                cohesion_bits.append("Mood targets: " + ", ".join(mood_set))
            lines.append("Cohesion Goals: " + " | ".join(cohesion_bits))
        
        if analysis_result.layout_slots:
            for slot in analysis_result.layout_slots:
                lines.append(
                    f"Slot '{slot.slot_name}': Source Image [{slot.source_images}], "
                    f"Pos x=[{slot.position.x}]%, y=[{slot.position.y}]%, "
                    f"Content: {slot.description}"
                )
        else:
            lines.append("No explicit layout slots provided; keep subjects on far left/right for front, emblem on lower back.")
        
        return "\n".join(lines)

    def _example_reference_parts(self) -> list:
        parts = []
        parts.append(types.Part.from_text(
            text="--- LEARNING FROM EXAMPLES ---\n"
            "Here are examples of how to convert Source Components into a Final Texture Map.\n"
            "Observe how the characters are placed to fit the layout, and how the background fills the rest."
        ))

        for idx, (texture_path, preview_path, note) in enumerate(self.example_pairs, start=1):
            if not os.path.exists(texture_path):
                continue
            
            # Check for component slots in assets/dspy_inputs
            try:
                base_name = os.path.basename(texture_path)
                ref_id = base_name.split('-')[0]
                
                slot_files = []
                for i in range(3):
                    slot_path = f"assets/dspy_inputs/{ref_id}-b_slot{i}.png"
                    if os.path.exists(slot_path):
                        slot_files.append(slot_path)
                
                if slot_files:
                    parts.append(types.Part.from_text(
                        text=f"\n=== EXAMPLE {idx} ==="
                    ))
                    parts.append(types.Part.from_text(
                        text="INPUT COMPONENTS (The raw materials):"
                    ))
                    for i, slot_path in enumerate(slot_files):
                        parts.append(self._image_part_from_path(slot_path))
                    
                    parts.append(types.Part.from_text(
                        text="OUTPUT TEXTURE MAP (The result):"
                    ))
                    parts.append(self._image_part_from_path(texture_path))
                    parts.append(types.Part.from_text(
                        text="Notice how the components are arranged to fill the canvas, with the main characters in the 'White Zones' of the mask."
                    ))
                    continue # Skip the standard adding if we did the detailed one
            except Exception as e:
                print(f"Error processing slots for example {idx}: {e}")

            # Fallback if no slots found
            parts.append(types.Part.from_text(
                text=(
                    f"Example {idx} Target Texture: '{texture_path}'."
                )
            ))
            parts.append(self._image_part_from_path(texture_path))

        return parts

    def _build_generation_parts(self, analysis_result: AnalysisResult, reference_images: list[str], mask_path: str = None) -> list:
        layout_lines = [
            f"LAYOUT PLAN (Follow Coordinates Exactly):",
            f"CONCEPTUAL DIVIDER (Do NOT draw this line): y={analysis_result.front_back_divider_y}% (Top=Front, Bottom=Back)."
        ]
        if analysis_result.layout_slots:
            for slot in analysis_result.layout_slots:
                layout_lines.append(
                    f"- SLOT '{slot.slot_name}': Place content from Image {slot.source_images} at x={slot.position.x}%, y={slot.position.y}%."
                )
        else:
            layout_lines.append("No explicit slots. Place Image 1 on Left, Image 2 on Right.")

        parts = [
            types.Part.from_text(
                text="TASK: Generate a 'Vinyl Skin Texture Map' for a handheld console.\n"
                "OUTPUT FORMAT: A single flat, 2D, square image (1:1 aspect ratio).\n"
                "STYLE: Anime/Illustration. High quality, seamless composition."
            ),
            types.Part.from_text(
                text="CRITICAL: This is a TEXTURE FILE, not a photograph of a device.\n"
                "- Do NOT render a 3D object.\n"
                "- Do NOT render the console casing, buttons, or screen bezels.\n"
                "- Do NOT draw any guidelines, grid lines, or divider lines.\n"
                "- The image should look like a flat sheet of paper with artwork printed on it."
            ),
                types.Part.from_text(
                    text="PROHIBITED CONTENT:\n"
                    "- Never write text such as 'Switch', 'Lite', 'Switchlite', or any hardware logos.\n"
                    "- Do not draw hardware silhouettes, button labels, or helper guide lines.\n"
                    "- The canvas must look like finished art without technical markings."
                ),
                types.Part.from_text(
                    text="COHESION PROTOCOL:\n"
                    "- Treat all source images as belonging to the same illustration.\n"
                    "- Normalize lighting, rendering quality, and brushwork so no character looks cut-out.\n"
                    "- Use the shared palette from the analysis summary to create a single gradient or pattern that spans the entire canvas (top+bottom).\n"
                    "- If colors clash, gently recolor accessories/backgrounds while preserving character identity."
                ),
            types.Part.from_text(
                text="INPUT 1: THE MASK (Geometry Guide)\n"
                "I will provide a black-and-white/grey mask image.\n"
                "- WHITE ZONES: These are the 'Safe Areas'. You MUST place the main characters/faces here.\n"
                "- GREY ZONES: These are 'Bleed/Cut Areas'. Fill these with background colors/patterns. Do NOT put important details here.\n"
                "- The Mask is a GUIDE. Do not draw the mask itself in the final output. Draw the ARTWORK."
            ),
            types.Part.from_text(
                text="INPUT 2: SOURCE IMAGES (The Content)\n"
                "I will provide reference images containing characters.\n"
                "- You must EXTRACT the main character from each source image.\n"
                "- You must PLACE that character into the specific 'White Zone' defined in the Layout Plan.\n"
                "- Do not change the character's design. Copy it as faithfully as possible."
            ),
            types.Part.from_text(text="\n".join(layout_lines)),
            types.Part.from_text(text=self._analysis_summary_text(analysis_result)),
        ]
        
        # Attach template mask
        current_mask_path = mask_path if mask_path else self.template_mask_path
        if current_mask_path and os.path.exists(current_mask_path):
            parts.append(types.Part.from_text(
                text="HERE IS THE MASK (Geometry Guide):"
            ))
            try:
                parts.append(self._image_part_from_path(current_mask_path))
            except Exception as mask_err:
                print(f"Warning: Failed to load template mask: {mask_err}")

        # Attach example pairs
        parts.extend(self._example_reference_parts())

        parts.append(types.Part.from_text(
            text="NOW, GENERATE THE TEXTURE FOR THE FOLLOWING INPUTS:"
        ))
        
        for idx, img_path in enumerate(reference_images):
            try:
                img_part = self._image_part_from_path(img_path)
            except Exception as e:
                continue
            
            parts.append(types.Part.from_text(
                text=(
                    f"SOURCE IMAGE {idx+1}:"
                )
            ))
            parts.append(img_part)

        # Load optimized instructions if available
        instructions_text = (
            "INSTRUCTIONS:\n"
            "1. Look at Source Image 1. Extract the character.\n"
            "2. Look at the Layout Plan. Find where Source Image 1 goes.\n"
            "3. Paint that character into that position on the canvas.\n"
            "4. Repeat for all source images.\n"
            "5. Fill the rest of the canvas (especially the Grey Zones of the mask) with a matching background pattern.\n"
            "6. OUTPUT: The final flat texture map."
        )
        
        optimized_instructions_path = "assets/optimized_instructions.txt"
        if os.path.exists(optimized_instructions_path):
            try:
                with open(optimized_instructions_path, 'r') as f:
                    custom_instructions = f.read().strip()
                if custom_instructions:
                    print(f"Using optimized instructions from {optimized_instructions_path}")
                    instructions_text = custom_instructions
            except Exception as e:
                print(f"Failed to load optimized instructions: {e}")

        parts.append(types.Part.from_text(text=instructions_text))
        return parts

    def generate_image(self, analysis_result: AnalysisResult, reference_images: list[str], output_path: str, mask_path: str = None):
        """
        Generates the actual image using the requested model based on the prompt and reference images.
        Uses generate_content with response_modalities=["IMAGE"].
        """
        try:
            # For Gemini 3 Pro Texture Generator, we use generate_content
            # asking for an image response modality.
            
            parts = self._build_generation_parts(analysis_result, reference_images, mask_path=mask_path)

            print(f"Sending prompt and {len(reference_images)} reference images to Image Model ({self.image_model})...")
            
            # Explicitly specify role as 'user' for the prompt content
            # Use GenerateContentConfig with aspect_ratio provided via prompt primarily, 
            # but try to pass standard generation params.
            # Note: The user suggested types.GenerateImagesConfig but that is for the older/imagen API.
            # For generate_content (gemini 3), we use GenerateContentConfig.
            # Based on model_fields inspection, 'aspect_ratio' is NOT a direct field of GenerateContentConfig.
            # However, 'image_config' IS a field.
            # So correct usage should be config=types.GenerateContentConfig(image_config=...)
            
            # Let's try to construct the image_config object if possible, or pass as dict if SDK allows.
            # The field name is 'image_config'.
            
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=[types.Content(
                    role="user",
                    parts=parts
                )],
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.IMAGE],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1"
                    )
                )
            )

            
            # Extract image from response parts
            # Look for inline_data
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img_data = part.inline_data.data

                        img = Image.open(BytesIO(img_data))
                        img.save(output_path)
                        return output_path

            print("No image found in response candidates. Response might be text only or blocked.")
            # Debug: print response structure if needed
            # print(response)
            return None

        except Exception as e:
            print(f"Image generation error: {e}")
            return None
