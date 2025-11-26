import os
import sys
import argparse
import json
from dotenv import load_dotenv
from src.services.vision import VisionService
from src.services.generation import GenerationService
from src.utils.image_ops import overlay_template

# Add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    parser = argparse.ArgumentParser(description="Nintendo Switch Lite Itasha Generator")
    parser.add_argument("image_paths", nargs='+', help="Path(s) to the input image(s) (element/style reference)")
    parser.add_argument("--output", default="assets/output/result.png", help="Path to save the final result")
    
    args = parser.parse_args()
    
    # Validate images
    valid_paths = []
    for path in args.image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Input file '{path}' not found. Skipping.")
    
    if not valid_paths:
        print("Error: No valid input images found.")
        return

    print(f"--- Initializing Services ---")
    try:
        vision = VisionService()
        generator = GenerationService()
    except Exception as e:
        print(f"Error initializing services: {e}")
        print("Please check your .env file and API keys.")
        return

    print(f"--- Step 1: Analyzing Input Images ({len(valid_paths)} files) ---")
    try:
        analysis_result = vision.analyze_image(valid_paths)
        
        # Pretty print the analysis result
        print(f"Analysis Result (JSON Structure):")
        print(analysis_result.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

    print(f"--- Step 2: Generating Design Prompt ---")
    try:
        design_prompt = generator.generate_design_prompt(analysis_result)
        print(f"Generated Prompt:\n{design_prompt}")
    except Exception as e:
        print(f"Error during prompt generation: {e}")
        return

    print(f"--- Step 3: Generating Image (Texture) ---")
    # Intermediate file
    temp_output = "assets/output/temp_gen.png"
    os.makedirs("assets/output", exist_ok=True)
    
    result_path = generator.generate_image(design_prompt, temp_output)
    
    if result_path:
        print(f"--- Step 4: Applying Template Overlay ---")
        overlay_template(result_path, args.output)
        print(f"Success! Final design saved to {args.output}")
    else:
        print("Failed to generate image.")

if __name__ == "__main__":
    main()
