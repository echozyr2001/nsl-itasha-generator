import argparse
import json
import os
from pathlib import Path
from PIL import Image
from typing import List, Tuple

# Define relative crop boxes (left, top, right, bottom) in percentage coordinates.
# These are heuristics to grab left hero, right hero, and bottom motifs from each texture.
DEFAULT_CROPS: List[Tuple[float, float, float, float]] = [
    (0.00, 0.00, 0.45, 0.55),  # upper-left quadrant
    (0.55, 0.00, 1.00, 0.55),  # upper-right quadrant
    (0.15, 0.55, 0.85, 1.00),  # lower central band
]

def extract_crops(texture_path: Path, output_dir: Path, crop_defs=None):
    crop_defs = crop_defs or DEFAULT_CROPS
    img = Image.open(texture_path)
    w, h = img.size
    crops = []
    for idx, (x0, y0, x1, y1) in enumerate(crop_defs):
        box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        crop = img.crop(box)
        out_path = output_dir / f"{texture_path.stem}_slot{idx}.png"
        crop.save(out_path)
        crops.append({
            "slot": idx,
            "box": box,
            "path": str(out_path.relative_to(output_dir.parent))
        })
    return crops

def main():
    parser = argparse.ArgumentParser(description="Extract heuristic reference crops from texture outputs.")
    parser.add_argument("texture_glob", nargs="?", default="ref/*-b.JPG")
    parser.add_argument("--output-dir", default="assets/dspy_inputs")
    parser.add_argument("--manifest", default="datasets/dspy_crops.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for path in sorted(Path().glob(args.texture_glob)):
        crops = extract_crops(path, output_dir)
        manifest.append({
            "texture": str(path),
            "crops": crops
        })
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
