from PIL import Image

def overlay_template(generated_image_path: str, output_path: str, mask_path: str = None):
    """
    Overlays the generated design onto a Switch Lite template.
    Currently just resizes/crops as a placeholder since we don't have the actual mask asset.
    """
    try:
        # Switch Lite dimensions approx 208mm x 91mm. 
        # Let's assume 1920x1080 for the generated image and crop to 2:1 approx.
        
        img = Image.open(generated_image_path)
        
        if mask_path and os.path.exists(mask_path):
            # If we have a mask, we could apply it here
            # For now, just saving the image as is, or maybe composite if mask has transparency
            pass
        
        # Placeholder: Just save it as the "final" for now.
        # In a real app, we would load a transparent PNG mask of the Switch Lite
        # and composite it over this image.
        
        img.save(output_path)
        print(f"Processed template saved to {output_path}")
        
    except Exception as e:
        print(f"Error in overlay: {e}")

