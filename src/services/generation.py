import os
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
# We import AnalysisResult only for type hinting if needed, but to be safe with circulars (even if none now)
# we can just assume the object structure or use Any.
# For clarity, let's import it.
from src.services.vision import AnalysisResult

load_dotenv()

class GenerationService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=self.api_key)
        # User requested Gemini 3 Pro for the "Control" / Logic part
        self.logic_model = "gemini-3-pro-preview"
        
        # User requested Gemini 3 Pro Image Preview for generation
        self.image_model = "gemini-3-pro-image-preview"

    def generate_design_prompt(self, analysis_result: AnalysisResult, user_preferences: str = "") -> str:
        """
        Uses Gemini 3 Pro (logic model) to create a precise image generation prompt
        based on the structured visual analysis and Switch Lite template constraints.
        """
        
        system_instruction = """
        You are a specialist in generative art for product skins.
        Your goal is to take a structured visual analysis of user-provided images and create a highly detailed
        text-to-image prompt that will generate a seamless pattern or composition suitable for a Nintendo Switch Lite skin.
        
        The Switch Lite has a specific aspect ratio (roughly 2:1 width to height) and control layouts.
        The design should be a "texture" or "wallpaper" style that can be overlaid on the device.
        Avoid generating buttons or screen elements; generate the ARTWORK that goes UNDER/AROUND them.
        
        INTEGRATE ELEMENTS FROM ALL INPUT IMAGES based on the analysis.
        
        Output ONLY the prompt string in English.
        """
        
        # Serialize the structured data to a clear text format for the LLM
        analysis_text = "--- Visual Analysis of Input Images ---\n"
        for i, img_analysis in enumerate(analysis_result.images):
            analysis_text += f"Image {i+1}:\n"
            analysis_text += f"  Description: {img_analysis.description}\n"
            analysis_text += f"  Elements: {', '.join(img_analysis.elements)}\n"
            analysis_text += f"  Style: {img_analysis.style}\n"
            analysis_text += f"  Colors: {', '.join(img_analysis.colors)}\n"
            analysis_text += f"  Mood: {img_analysis.mood}\n\n"
        
        analysis_text += f"--- Synthesis & Suggestion ---\n{analysis_result.synthesis}\n"

        content = f"""
        {analysis_text}
        
        User Preferences (if any):
        {user_preferences}
        
        Create the prompt.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.logic_model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=content)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            return response.text.strip()
        except Exception as e:
            # Fallback if the specific model fails, try a standard one
            print(f"Warning: Logic model failed ({e}), falling back to gemini-2.0-flash")
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[content] # Simplified for fallback
                )
                return response.text.strip()
            except Exception as e2:
                return f"Error generating prompt: {str(e2)}"

    def generate_image(self, prompt: str, output_path: str):
        """
        Generates the actual image using the requested model based on the prompt.
        Uses generate_content with response_modalities=["IMAGE"].
        """
        try:
            # For Gemini 3 Pro Image Preview, we use generate_content
            # asking for an image response modality.
            
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )
            
            # Extract image from response parts
            # Look for inline_data
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Assuming bytes data is directly accessible or handled by SDK
                        # In some SDK versions, part.inline_data.data is bytes or base64 string
                        # The google-genai SDK typically exposes .data as bytes for decoded content if helper used
                        # or we might need to decode. 
                        # However, Image.open accepts bytes-like object.
                        
                        img_data = part.inline_data.data
                        
                        img = Image.open(BytesIO(img_data))
                        img.save(output_path)
                        return output_path
            
            print("No image found in response candidates.")
            return None

        except Exception as e:
            print(f"Image generation error: {e}")
            return None
