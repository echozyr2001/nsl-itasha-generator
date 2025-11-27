import os
import sys
import argparse
from dotenv import load_dotenv
from src.services.vision import VisionService
from src.services.generation import GenerationService
from src.utils.image_ops import overlay_template

# Add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def main():
    parser = argparse.ArgumentParser(description="Nintendo Switch Lite Itasha Generator")
    parser.add_argument("image_paths", nargs='+', help="Path(s) to the input image(s) (element/style reference)")
    parser.add_argument("--output", default="assets/output/result.png", help="Path to save the generated base texture (no mask applied)")
    parser.add_argument("--preview-output", help="Optional path to save a preview with the Switch Lite mask applied")
    
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

    print(f"--- Step 2: Generating Image (Texture) ---")
    # Intermediate file
    temp_output = "assets/output/temp_gen.png"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Pass analysis result and images directly to the image generator
    base_texture_path = generator.generate_image(analysis_result, valid_paths, args.output)
    
    if not base_texture_path:
        print("Failed to generate image.")
        return
    
    print(f"Base texture saved to {args.output}")
    
    if args.preview_output:
        os.makedirs(os.path.dirname(args.preview_output), exist_ok=True)
        print(f"--- Optional Preview: Applying Template Overlay ---")
        overlay_template(base_texture_path, args.preview_output)
        print(f"Preview with mask saved to {args.preview_output}")

if __name__ == "__main__":
    main()
