"""
NLP Fine-Tuning Data Generator — Creates prompt/response training pairs
from real floor plan datasets for fine-tuning the Ollama SLM.

Generates diverse natural language prompts that describe floor plans
from CubiCasa5K and Modified Swiss Dwellings, paired with the expected
JSON output matching our ParsedLayout schema.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Optional

from app.models.schemas import ParsedLayout, RoomType

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    # BHK style
    "{bhk}BHK {style} apartment with {features}, {facing} facing",
    "{bhk}BHK flat, {facing} facing, {features}",
    "Design a {bhk}BHK {style} home with {features}",
    "I need a {facing} facing {bhk}BHK with {features}, {area} sqft",
    "{bhk} bedroom {style} house, {features}, {facing} entrance",

    # Descriptive style
    "A {facing} facing home with {room_list}",
    "Create a floor plan with {room_list}, total {area} sqft, {facing} facing",
    "{style} {facing}-facing apartment: {room_list}",
    "Design a {area} sqft home with {room_list}",
    "Plan a {style} {facing} facing residence with {room_list}",

    # Indian terminology
    "{bhk}BHK with modular kitchen and attached bathrooms, {facing} facing",
    "Vastu-compliant {bhk}BHK, {facing} facing, pooja room, {features}",
    "{bhk}BHK {style} flat, {area} sqft, {facing} facing, {features}",
    "Draw a {bhk}BHK layout, hall-kitchen open, {facing} entrance",
    "{bhk}BHK with drawing room, {features}, {facing} facing, {area} sqft",
]

STYLES = [
    "modern", "contemporary", "traditional", "minimalist", "open-concept",
    "spacious", "compact", "luxury", "cozy", "smart", "premium",
]

FEATURES = [
    "open kitchen", "attached bathrooms", "balcony", "study room",
    "modular kitchen", "walk-in closet", "utility area",
    "separate dining", "large balcony", "pooja room", "store room",
    "car parking", "servant room", "wash area", "terrace garden",
]

FACINGS = ["north", "south", "east", "west"]

ROOM_TYPE_LABELS = {
    RoomType.LIVING_ROOM: ["living room", "hall", "drawing room", "sitting room"],
    RoomType.MASTER_BEDROOM: ["master bedroom", "main bedroom", "master suite"],
    RoomType.BEDROOM: ["bedroom", "bed room", "guest room"],
    RoomType.KITCHEN: ["kitchen", "modular kitchen", "open kitchen"],
    RoomType.BATHROOM: ["bathroom", "attached bathroom", "common bathroom"],
    RoomType.TOILET: ["toilet", "WC", "powder room"],
    RoomType.CORRIDOR: ["corridor", "hallway", "passage"],
    RoomType.BALCONY: ["balcony", "verandah", "terrace"],
    RoomType.STUDY: ["study", "study room", "pooja room", "office"],
    RoomType.DINING: ["dining", "dining room", "dining area"],
    RoomType.UTILITY: ["utility", "wash area", "store room", "laundry"],
    RoomType.GARAGE: ["garage", "car parking", "parking"],
}


def _count_bedrooms(rooms: list[dict]) -> int:
    """Count bedrooms (BEDROOM + MASTER_BEDROOM) in room list."""
    bedroom_types = {"BEDROOM", "MASTER_BEDROOM"}
    return sum(1 for r in rooms if r.get("room_type", "") in bedroom_types)


def _generate_prompt_from_layout(layout_dict: dict) -> str:
    """Generate a natural language prompt from a parsed layout dict."""
    rooms = layout_dict.get("rooms", [])
    facing = layout_dict.get("facing", "NORTH").lower()
    area_sqm = layout_dict.get("plot_area_sqm", 100)
    area_sqft = round(area_sqm * 10.764)

    bhk = _count_bedrooms(rooms)

    # Build room description list
    room_descriptions = []
    for room in rooms:
        rt = room.get("room_type", "ROOM")
        labels = ROOM_TYPE_LABELS.get(RoomType(rt), [rt.lower().replace("_", " ")])
        label = random.choice(labels)
        area = room.get("target_area_sqm", 0)
        if area > 0 and random.random() < 0.3:
            label += f" ({area:.0f} sqm)"
        room_descriptions.append(label)

    room_list = ", ".join(room_descriptions)

    # Pick random features
    feature_count = random.randint(1, 3)
    features = ", ".join(random.sample(FEATURES, min(feature_count, len(FEATURES))))

    style = random.choice(STYLES)

    # Pick template
    template = random.choice(PROMPT_TEMPLATES)

    try:
        prompt = template.format(
            bhk=bhk,
            style=style,
            features=features,
            facing=facing,
            area=area_sqft,
            room_list=room_list,
        )
    except (KeyError, IndexError):
        prompt = f"{bhk}BHK {style} apartment, {facing} facing, {area_sqft} sqft, with {room_list}"

    return prompt


def generate_nlp_training_data(
    dataset_samples: list[dict],
    output_path: str = "data/training/nlp_finetune.jsonl",
    augmentation_factor: int = 3,
) -> str:
    """Generate NLP fine-tuning pairs from floor plan dataset samples.

    Each sample generates multiple prompt variations paired with
    the same expected JSON output.

    Args:
        dataset_samples: List of dicts with parsed_layout key
        output_path: Output JSONL path
        augmentation_factor: Number of prompt variants per sample

    Returns:
        Path to output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pairs = []
    for sample in dataset_samples:
        layout = sample.get("parsed_layout", {})
        if not layout.get("rooms"):
            continue

        # Generate multiple prompt variations
        for _ in range(augmentation_factor):
            prompt = _generate_prompt_from_layout(layout)
            pairs.append({
                "prompt": prompt,
                "response": json.dumps(layout, indent=2),
                "system": "nlp_parser",
            })

    random.shuffle(pairs)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Generated {len(pairs)} NLP fine-tuning pairs → {output_path}")
    return output_path


def generate_ollama_modelfile(
    base_model: str = "llama3.2:3b",
    training_data_path: str = "data/training/nlp_finetune.jsonl",
    output_path: str = "data/training/Modelfile",
) -> str:
    """Generate an Ollama Modelfile for fine-tuning with training pairs.

    Creates a Modelfile with system prompt + few-shot examples
    embedded as template conversations.

    Args:
        base_model: Base Ollama model name
        training_data_path: Path to JSONL training data
        output_path: Output Modelfile path

    Returns:
        Path to Modelfile
    """
    from app.services.nlp_parser import build_system_prompt

    system = build_system_prompt()

    # Load a few examples for the Modelfile template
    examples = []
    if os.path.exists(training_data_path):
        with open(training_data_path, "r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                    if len(examples) >= 5:
                        break

    modelfile = f"""FROM {base_model}

SYSTEM \"\"\"{system}\"\"\"

PARAMETER temperature 0.1
PARAMETER num_predict 1024
PARAMETER top_p 0.9
"""

    # Add example conversations as TEMPLATE
    for i, ex in enumerate(examples):
        modelfile += f"""
MESSAGE user {ex['prompt']}
MESSAGE assistant {ex['response']}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(modelfile)

    logger.info(f"Generated Ollama Modelfile → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate NLP fine-tuning data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "cubicasa", "swiss", "combined"])
    parser.add_argument("--output", type=str, default="data/training/nlp_finetune.jsonl")
    parser.add_argument("--augment", type=int, default=3, help="Prompt variants per sample")
    parser.add_argument("--modelfile", action="store_true", help="Also generate Ollama Modelfile")
    args = parser.parse_args()

    # Load samples
    from app.services.train_gnn import load_dataset_by_source
    samples = load_dataset_by_source(args.dataset, args.data_dir)

    output = generate_nlp_training_data(samples, args.output, args.augment)

    if args.modelfile:
        generate_ollama_modelfile(training_data_path=output)
