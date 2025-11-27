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
        lines = ["=== STRUCTURED ANALYSIS ==="]
        for idx, img in enumerate(analysis_result.images):
            lines.append(f"\nImage {idx+1}:")
            lines.append(f"  Description: {img.description}")
            lines.append(f"  Elements: {', '.join(img.elements)}")
            lines.append(f"  Style: {img.style}")
            lines.append(f"  Colors: {', '.join(img.colors)}")
            lines.append(f"  Mood: {img.mood}")
        
        lines.append(f"\n=== SYNTHESIS (CRITICAL: How to combine all images) ===")
        lines.append(f"{analysis_result.synthesis}")
        lines.append(f"\n=== CANVAS STRUCTURE ===")
        lines.append(f"Front/Back Divider: y={analysis_result.front_back_divider_y}% (front: 0-{analysis_result.front_back_divider_y}%, back: {analysis_result.front_back_divider_y}-100%)")
        
        if analysis_result.layout_slots:
            lines.append(f"\n=== LAYOUT SLOTS (exact placement plan) ===")
            for slot in analysis_result.layout_slots:
                lines.append(
                    f"\n{slot.slot_name}:"
                )
                lines.append(f"  Source images: {slot.source_images} (use elements from these reference images)")
                lines.append(f"  Position: x={slot.position.x}%, y={slot.position.y}%")
                lines.append(f"  Content: {slot.description}")
                lines.append(f"  Purpose: {slot.purpose}")
                lines.append(f"  Avoid: {slot.avoid}")
        else:
            lines.append("\n=== LAYOUT SLOTS ===")
            lines.append("No explicit layout slots provided; keep subjects on far left/right for front, emblem on lower back.")
        
        return "\n".join(lines)

    def _example_reference_parts(self) -> list:
        parts = []
        parts.append(types.Part.from_text(
            text="REFERENCE OUTPUT EXAMPLES (pre-mask textures only): ref/1-b.JPG, ref/2-b.JPG, ref/3-b.JPG, ref/4-b.JPG, ref/5-b.JPG, ref/6-b.JPG, ref/7-b.JPG, ref/8-b.JPG, ref/9-b.JPG."
        ))

        for idx, (texture_path, preview_path, note) in enumerate(self.example_pairs, start=1):
            if not os.path.exists(texture_path):
                continue
            parts.append(types.Part.from_text(
                text=(
                    f"Example Pair {idx}: '{texture_path}' is the TARGET printable vinyl texture (what you must create). "
                    "Study how it reuses reference art directly."
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
        layout_lines = [
            f"Layout plan derived from structured analysis (percent coords, 0-100):",
            f"IMPORTANT: Front/Back divider is at y={analysis_result.front_back_divider_y}% (NOT 50%). "
            f"Front section: y=0% to y={analysis_result.front_back_divider_y}%. Back section: y={analysis_result.front_back_divider_y}% to y=100%."
        ]
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
                text="You are generating the RAW printable vinyl texture (same deliverable as ref/1-b.JPG ... ref/9-b.JPG). "
                "Return a full-color square artwork only—no mask overlay, no grey filler, and no hardware silhouettes."
            ),
            types.Part.from_text(
                text="Render only a flat, front-facing printable vinyl sheet. "
                "Keep the viewpoint orthographic (straight-on) with no perspective, bevels, shadows, lighting cues, plastic edges, logos, or button labels."
            ),
            types.Part.from_text(
                text="Maintain a HARD SEPARATION between the front (top) and back (bottom) halves. "
                "All characters and focal motifs assigned to the front MUST stay entirely above the divider line; "
                "subjects dedicated to the back must stay entirely below it. "
                "Allow only background gradients to cross the split so the two halves do not bleed into each other."
            ),
            types.Part.from_text(
                text="Use the mask only for proportions. Top/front white band spans roughly y:3-54%, bottom/back spans y:57-97%—match this ratio."
            ),
            types.Part.from_text(
                text="REFERENCE FIDELITY RULES: You are not inventing new characters. "
                "Treat the supplied references as mandatory key art that must be rotoscoped into the final layout. "
                "Hair color, face shape, outfit silhouettes, weapons, decals, and lighting accents must match the reference pixel-for-pixel (minor pose tweaks only to fit the slot). "
                "If a reference shows a twin-tailed girl with a specific rifle, that exact girl and rifle must appear—do NOT replace them with a "
                "\"similar\" design or generic interpretation."
            ),
            types.Part.from_text(
                text="The final art must look like the provided TARGET OUTPUT examples (2-b.JPG and 8-b.JPG): a continuous illustration across a 1:1 canvas with no hardware outlines, faux shadows, or transparent voids."
            ),
            types.Part.from_text(
                text="All placement decisions come from the preceding analysis plan. Follow every layout_slot coordinate precisely. "
                "If a slot reserves x:[10,25], y:[60,82], ensure the subject fully occupies that rectangle."
            ),
            types.Part.from_text(
                text="Front (top) design guidance: concentrate the rifle barrel and torsos along the left and right grips, keep the center screen band mostly background glow so the user's cutout will not remove faces or weapons."
            ),
            types.Part.from_text(
                text="The mask's white regions are the only areas that survive after fabrication. Concentrate characters, emblems, and important motifs fully inside those coordinates. Use grey-zone areas solely for low-contrast background gradients that can be safely trimmed away."
            ),
            types.Part.from_text(
                text="Do NOT hint at the mask outline: avoid blurred borders, ghosted rectangles, semi-transparent overlays, or softened screen silhouettes. Keep color and detail consistent across cut-line boundaries so the texture looks pristine before masking."
            ),
            types.Part.from_text(
                text="Top-half focal avoidance: keep the center window (x=25-75, y=18-50) to gradients/energy, never place faces there. Use the corners at x<20 and x>80 for the major guns/heads as shown in the layout slots."
            ),
            types.Part.from_text(
                text=f"CRITICAL DIVIDER ALIGNMENT: The mask's front/back split is NOT at 50%. "
                f"Use y={analysis_result.front_back_divider_y}% as the exact divider (front ≈ y:0-{analysis_result.front_back_divider_y}, back ≈ y:{analysis_result.front_back_divider_y}-100). "
                "Treat the top band as slimmer and the bottom as deeper; proportion the artwork just like ref/2-b.JPG."
            ),
            types.Part.from_text(
                text="Forbidden zones: Top-front screen area (x=25%-75%, y=18%-50%) and controller holes must contain only background gradients. Keep all faces and focal elements outside those bounds."
            ),
            types.Part.from_text(
                text="Screen avoidance rule: in the front half, keep the central rectangle (x=25-75, y=18-50) mostly background so the user can punch out the screen without losing character faces or weapons."
            ),
            types.Part.from_text(
                text="Mask context only: the provided mask image is for measuring coordinates. Do NOT copy its greys, do NOT embed transparent windows, and do NOT show outlines of buttons or vents."
            ),
            types.Part.from_text(text="\n".join(layout_lines)),
            types.Part.from_text(text=self._analysis_summary_text(analysis_result)),
        ]

        # Add explicit multi-image combination guidance
        if len(reference_images) > 1:
            parts.append(types.Part.from_text(
                text=f"CRITICAL MULTI-IMAGE INSTRUCTION: You have received {len(reference_images)} reference images. "
                f"The 'Synthesis' section above explains HOW to combine elements from all images. "
                f"The 'Layout Slots' section specifies EXACTLY which elements from which reference image go into which position. "
                f"You MUST follow this plan precisely: "
                f"1. Read each layout slot's 'Source images' field to know which reference image(s) to use for that slot. "
                f"2. Extract ONLY the elements described in that slot's 'Content' field from the specified reference image(s). "
                f"3. Place those elements at the exact coordinates given in the slot's 'Position' field. "
                f"4. Maintain visual consistency: if the same character appears in multiple slots, ensure it looks identical across all instances. "
                f"5. DO NOT simply copy one reference image or blend them randomly—create a unified composition by following the layout plan exactly. "
                f"6. You are required to literally reuse the referenced elements (copy the character, clothing, weapons, facial expression). Text-only reinterpretations are forbidden."
            ))

        identity_lines = [
            "REFERENCE IDENTITY LOCK (replicate these traits exactly; reuse the literal characters/weapons from the photos)."
        ]
        for idx, img_info in enumerate(analysis_result.images):
            elements = ", ".join(img_info.elements)
            colors = ", ".join(img_info.colors)
            identity_lines.append(
                f"Image {idx+1}: {img_info.description}. Preserve elements [{elements}] with palette [{colors}] and mood '{img_info.mood}'."
            )
        parts.append(types.Part.from_text(text="\n".join(identity_lines)))

        # Attach template mask to give spatial context
        if self.template_mask_path:
            parts.append(types.Part.from_text(
                text="Template mask placement aid ONLY: white = survives after cutting, grey = removed. "
                "Do NOT reproduce the mask graphic itself or its flat grey background—return a full-color printable texture instead."
            ))
            try:
                parts.append(self._image_part_from_path(self.template_mask_path))
            except Exception as mask_err:
                print(f"Warning: Failed to load template mask ({self.template_mask_path}): {mask_err}")

        # Attach example pairs for few-shot guidance
        parts.extend(self._example_reference_parts())

        # Add a summary of which elements should come from which image
        if len(reference_images) > 1:
            slot_to_image_map = {}
            if analysis_result.layout_slots:
                for slot in analysis_result.layout_slots:
                    for img_idx in slot.source_images:
                        if img_idx not in slot_to_image_map:
                            slot_to_image_map[img_idx] = []
                        slot_to_image_map[img_idx].append(slot.slot_name)
            
            if slot_to_image_map:
                mapping_text = "Element-to-Image Mapping (follow this exactly):\n"
                for img_idx, slots in slot_to_image_map.items():
                    if img_idx <= len(reference_images):
                        mapping_text += f"  - Reference Image {img_idx}: Use elements in slots {', '.join(slots)}\n"
                parts.append(types.Part.from_text(text=mapping_text))
        
        for idx, img_path in enumerate(reference_images):
            try:
                img_part = self._image_part_from_path(img_path)
            except Exception as e:
                print(f"Warning: Failed to load reference image {img_path} for generation: {e}")
                continue
            description = analysis_result.images[idx].description if idx < len(analysis_result.images) else "User reference."
            
            # Find which slots use this image
            relevant_slots = []
            if analysis_result.layout_slots:
                for slot in analysis_result.layout_slots:
                    if (idx + 1) in slot.source_images:  # source_images is 1-based
                        relevant_slots.append(f"{slot.slot_name} ({slot.description})")
            
            slot_info = f" This image's elements should be used in: {', '.join(relevant_slots)}" if relevant_slots else ""
            
            parts.append(types.Part.from_text(
                text=(
                    f"Reference Image {idx+1}: '{img_path}'.{slot_info} "
                    f"Maintain its characters, outfits, color palette, and key motifs exactly—only reposition and restyle them per the layout plan. "
                    f"Preserve identifiable features (faces, clothing, props) exactly as shown in this source image."
                )
            ))
            parts.append(types.Part.from_text(text=description))
            parts.append(img_part)

        parts.append(types.Part.from_text(
            text="REMINDER: Output a clean, guide-free base texture. Do not overlay the mask yourself, and do not leave any grey canvas."
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
