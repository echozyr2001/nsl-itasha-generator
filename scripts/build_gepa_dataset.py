import json
from pathlib import Path

DIVIDER_Y = 52.5
FRONT_Y = [0.0, DIVIDER_Y]
BACK_Y = [DIVIDER_Y + 4.0, 97.0]

FRONT_LEFT = {
    "slot_name": "Front-Left Hero",
    "position": {"x": [0.0, 45.0], "y": FRONT_Y},
    "purpose": "Primary subject hugging the left grip",
    "avoid": "screen cutout, D-pad"
}
FRONT_RIGHT = {
    "slot_name": "Front-Right Hero",
    "position": {"x": [55.0, 100.0], "y": FRONT_Y},
    "purpose": "Supporting hero on right grip",
    "avoid": "screen cutout, ABXY"
}
BACK_MAIN = {
    "slot_name": "Back Center",
    "position": {"x": [10.0, 90.0], "y": BACK_Y},
    "purpose": "Large motif for back plate",
    "avoid": "edge cutouts"
}

SLOT_TEMPLATES = [FRONT_LEFT, FRONT_RIGHT, BACK_MAIN]


def main():
    manifest_path = Path("datasets/dspy_crops.json")
    data = json.loads(manifest_path.read_text())
    dataset = []
    for entry in data:
        texture = entry["texture"]
        crops = entry["crops"]
        images = []
        layout_slots = []
        for idx, crop in enumerate(crops):
            slot_template = SLOT_TEMPLATES[idx]
            images.append({
                "description": f"Reference crop {idx} for {texture}",
                "elements": [f"slot{idx}"],
                "style": "unknown",
                "colors": [],
                "mood": "",
            })
            slot = {
                **slot_template,
                "source_images": [idx + 1],
                "description": f"Place crop {idx} content here",
            }
            layout_slots.append(slot)
        analysis = {
            "images": images,
            "synthesis": "Combine the reference crops using the predefined layout slots.",
            "layout_slots": layout_slots,
            "front_back_divider_y": DIVIDER_Y,
        }
        dataset.append({
            "texture": texture,
            "analysis": analysis,
            "references": [crop["path"] for crop in crops],
        })
    out_path = Path("datasets/gepa_dataset.json")
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
