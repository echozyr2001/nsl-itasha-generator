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
    def __init__(self):
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
            ("ref/2-b.JPG", "ref/2-a.JPG", "Example pair demonstrating cheerful mascot layout; ref/2-b.JPG is the printable texture, ref/2-a.JPG shows how it appears after applying the mask."),
            ("ref/8-b.JPG", "ref/8-a.JPG", "Example pair showing chibi characters with soft background; ref/8-b.JPG is the printable texture, ref/8-a.JPG is the masked preview."),
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
        lines = ["StructuredAnalysis:"]
        for idx, img in enumerate(analysis_result.images):
            lines.append(f"Image {idx+1}:")
            lines.append(f"  Description: {img.description}")
            lines.append(f"  Elements: {', '.join(img.elements)}")
            lines.append(f"  Style: {img.style}")
            lines.append(f"  Colors: {', '.join(img.colors)}")
            lines.append(f"  Mood: {img.mood}")
        lines.append(f"Synthesis: {analysis_result.synthesis}")
        if analysis_result.layout_slots:
            lines.append("LayoutSlots (percent coords relative to full canvas):")
            for slot in analysis_result.layout_slots:
                lines.append(
                    f"  - {slot.slot_name}: from images {slot.source_images}, "
                    f"x={slot.position.x}, y={slot.position.y}. "
                    f"Purpose: {slot.purpose}. Content: {slot.description}. Avoid: {slot.avoid}"
                )
        return "\n".join(lines)

    def _example_reference_parts(self) -> list:
        parts = []
        parts.append(types.Part.from_text(
            text="IMPORTANT: Synthesize a new 2D illustration; do not paste or collage the reference textures."
        ))

        for idx, (texture_path, preview_path, note) in enumerate(self.example_pairs, start=1):
            if not os.path.exists(texture_path):
                continue
            parts.append(types.Part.from_text(
                text=(
                    f"Example Pair {idx}: '{texture_path}' is the TARGET printable vinyl texture (what you must create)."
                )
            ))
            parts.append(types.Part.from_text(
                text=f"First, observe the TARGET texture '{texture_path}' (flat 2D artwork)."
            ))
            parts.append(self._image_part_from_path(texture_path))
            if os.path.exists(preview_path):
                parts.append(types.Part.from_text(
                    text=(
                        f"For context only: '{preview_path}' shows how that texture looks after the mask is applied. "
                        "Use it solely to understand which zones remain; ignore its hardware outlines when painting."
                    )
                ))
                parts.append(self._image_part_from_path(preview_path))
        return parts

    def _build_generation_parts(self, analysis_result: AnalysisResult, reference_images: list[str]) -> list:
        layout_lines = ["Layout plan derived from structured analysis (percent coords, 0-100):"]
        if analysis_result.layout_slots:
            for slot in analysis_result.layout_slots:
                layout_lines.append(
                    f"{slot.slot_name}: x={slot.position.x}, y={slot.position.y}. "
                    f"Purpose={slot.purpose}. Content={slot.description}. Avoid={slot.avoid}. "
                    f"Source images={slot.source_images}."
                )
        else:
            layout_lines.append("No explicit layout slots provided; keep subjects on far left/right for front, emblem on lower back.")

        parts = [
            types.Part.from_text(
                text="You are generating a base texture for a custom skin texture. "
                "This image must remain unmasked; the user will later overlay the provided mask to cut out screen/buttons. "
                "Do NOT paint guides, button icons, circular placeholders, or transparent holes. Treat the surface as a full canvas."
            ),
            types.Part.from_text(
                text="Render only a flat, front-facing printable vinyl sheet. "
                "Keep the viewpoint orthographic (straight-on) with no perspective, bevels, shadows, or lighting cues that imply hardware depth. "
                "Never depict the handheld console device, plastic edges, controllers, or any physical hardware body—only the 2D artwork that will be printed."
            ),
            types.Part.from_text(
                text="The final art must look like the provided TARGET OUTPUT examples (2-b.JPG and 8-b.JPG): a continuous illustration across a 1:1 canvas with no hardware outlines, faux shadows, or transparent voids."
            ),
            types.Part.from_text(
                text="All placement decisions come from the preceding analysis plan. Follow every layout_slot coordinate precisely. "
                "If a slot reserves x:[10,25], y:[60,82], ensure the subject fully occupies that rectangle."
            ),
            types.Part.from_text(
                text="The mask's white regions are the only areas that survive after fabrication. Concentrate characters, emblems, and important motifs fully inside those coordinates. Use grey-zone areas solely for low-contrast background gradients that can be safely trimmed away."
            ),
            types.Part.from_text(
                text="Do NOT hint at the mask outline: avoid blurred borders, ghosted rectangles, semi-transparent overlays, or softened screen silhouettes. Keep color and detail consistent across cut-line boundaries so the texture looks pristine before masking."
            ),
            types.Part.from_text(
                text="Top half = front half of the canvas. It must appear busier and more character-rich than the bottom half. Extend design elements all the way into the front-left and front-right corners so no blank wedges remain."
            ),
            types.Part.from_text(
                text="Forbidden zones: Top-front screen area (x=25%-75%, y=18%-50%) and controller holes must contain only background gradients. Keep all faces and focal elements outside those bounds."
            ),
            types.Part.from_text(text="\n".join(layout_lines)),
            types.Part.from_text(text=self._analysis_summary_text(analysis_result)),
        ]

        # Attach template mask to give spatial context
        if self.template_mask_path:
            parts.append(types.Part.from_text(
                text="Template mask placement aid ONLY: white = survives after cutting, grey = will be removed. "
                "Use it solely to position characters and patterns. Do not copy its shapes, colors, or transparency into the final artwork."
            ))
            try:
                parts.append(self._image_part_from_path(self.template_mask_path))
            except Exception as mask_err:
                print(f"Warning: Failed to load template mask ({self.template_mask_path}): {mask_err}")

        # Attach example pairs for few-shot guidance
        parts.extend(self._example_reference_parts())

        for idx, img_path in enumerate(reference_images):
            try:
                img_part = self._image_part_from_path(img_path)
            except Exception as e:
                print(f"Warning: Failed to load reference image {img_path} for generation: {e}")
                continue
            description = analysis_result.images[idx].description if idx < len(analysis_result.images) else "User reference."
            parts.append(types.Part.from_text(
                text=(
                    f"Reference Image {idx+1}: '{img_path}'. Maintain its characters, outfits, color palette, and key motifs exactly—only reposition and restyle them per the layout plan."
                )
            ))
            parts.append(types.Part.from_text(
                text=f"Use the subjects from Reference Image {idx+1}. Preserve identifiable features exactly as in the source."
            ))
            parts.append(types.Part.from_text(text=description))
            parts.append(img_part)

        parts.append(types.Part.from_text(
            text="REMINDER: Output a clean, guide-free base texture. Do not overlay the mask yourself."
        ))
        parts.append(types.Part.from_text(
            text="Final deliverable must be a seamless 1:1 canvas with zero transparent pixels and no visible hardware edges—only artwork ready for printing."
        ))
        parts.append(types.Part.from_text(
            text="Fill every region (including the future cut-out zones) with cohesive background gradients or patterns so that when the user applies the mask, only the white regions remain."
        ))
        parts.append(types.Part.from_text(
            text="Reject any impulse to outline the device frame, screen bezel, controls, D-pad, or buttons. Those shapes should only emerge later when the user applies the mask."
        ))
        parts.append(types.Part.from_text(
            text="Double-check the analysis layout: if a slot sits in the top half's white grip region, keep the center screen zone minimal. If a slot targets the back (lower half), fill that area generously so the final product feels rich after masking."
        ))
        parts.append(types.Part.from_text(
            text="Ensure the top half background remains a clean gradient or pattern with no boxed blur mirroring the screen area—treat it as uninterrupted art."
        ))
        parts.append(types.Part.from_text(
            text="Reinforce the front-left and front-right corners with motifs, gradients, or supporting elements so they do not look empty after masking."
        ))
        return parts

    def generate_image(self, analysis_result: AnalysisResult, reference_images: list[str], output_path: str):
        """
        Generates the actual image using the requested model based on the prompt and reference images.
        Uses generate_content with response_modalities=["IMAGE"].
        """
        try:
            # For Gemini 3 Pro Texture Generator, we use generate_content
            # asking for an image response modality.
            
            parts = self._build_generation_parts(analysis_result, reference_images)

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
