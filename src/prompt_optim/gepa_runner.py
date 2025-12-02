"""GEPA optimizer runner for prompt optimization."""
from __future__ import annotations

import json
import sys
import os
import random
from copy import deepcopy
from pathlib import Path

import dspy

from src.prompt_optim.prompt_composer import PromptComposer
from src.prompt_optim.eval_rules import score_prompt
from src.prompt_optim.dspy_config import configure_dspy

DATASET_PATH = Path("datasets/gepa_dataset.json")
OUTPUT_PATH = Path("datasets/gepa_optimized_prompt.txt")
MAX_TRAINSET_SIZE = int(os.getenv("GEPA_MAX_TRAINSET", "12"))


def load_dataset(include_mixed: bool = True):
    """Load the GEPA training dataset and optionally add interleaved examples."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    dataset = json.loads(DATASET_PATH.read_text())
    if include_mixed and dataset:
        mixed = build_interleaved_examples(dataset)
        if mixed:
            print(f"Augmenting dataset with {len(mixed)} mixed-reference examples")
            dataset = dataset + mixed
    return dataset


def build_interleaved_examples(dataset: list[dict]) -> list[dict]:
    """Generate synthetic examples that mix references from different base textures."""
    slot0_refs = [ex["references"][0] for ex in dataset if len(ex.get("references", [])) > 0]
    slot1_refs = [ex["references"][1] for ex in dataset if len(ex.get("references", [])) > 1]
    slot2_refs = [ex["references"][2] for ex in dataset if len(ex.get("references", [])) > 2]

    if not (slot0_refs and slot1_refs and slot2_refs):
        return []

    len0, len1, len2 = len(slot0_refs), len(slot1_refs), len(slot2_refs)
    shift1 = max(1, len1 // 2)
    shift2 = max(1, len2 // 3)

    mixed_examples: list[dict] = []
    for idx, base in enumerate(dataset):
        ref0 = slot0_refs[idx % len0]
        ref1 = slot1_refs[(idx + shift1) % len1]
        ref2 = slot2_refs[(idx + shift2) % len2]

        if len({ref0, ref1, ref2}) < 3:
            continue  # skip combos that accidentally reuse same image

        analysis_copy = deepcopy(base["analysis"])
        update_image_descriptor(analysis_copy, 0, ref0)
        update_image_descriptor(analysis_copy, 1, ref1)
        update_image_descriptor(analysis_copy, 2, ref2)

        mixed_examples.append(
            {
                "texture": base.get("texture"),
                "analysis": analysis_copy,
                "references": [ref0, ref1, ref2],
            }
        )

    return mixed_examples


def update_image_descriptor(analysis: dict, image_idx: int, ref_path: str) -> None:
    """Update analysis metadata to describe the new reference source."""
    images = analysis.get("images", [])
    if image_idx >= len(images):
        return
    descriptor = images[image_idx]
    descriptor["description"] = f"Use primary character from {ref_path}"
    descriptor["style"] = "Match original art style"
    descriptor["mood"] = "Preserve emotional tone"


def create_gepa_metric(dataset: list[dict]):
    """
    Create a GEPA metric function that generates images and evaluates them.
    Returns a function that conforms to GEPA's metric signature.
    """
    # Initialize image evaluator once (lazy loading to avoid initialization errors)
    evaluator = None
    
    def get_evaluator():
        nonlocal evaluator
        if evaluator is None:
            from src.prompt_optim.image_evaluator import ImageEvaluator
            evaluator = ImageEvaluator()
        return evaluator
    
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        GEPA metric function that generates images and evaluates them.
        
        Args:
            gold: The gold example (from trainset) - contains analysis_json and reference_paths
            pred: The predicted output (from program execution) - contains prompt
            trace: Optional trace of program execution
            pred_name: Optional name of predictor being optimized
            pred_trace: Optional trace of predictor execution
            
        Returns:
            float score based on actual image generation and evaluation
        """
        try:
            # Extract prompt from prediction
            prompt = pred.get("prompt", "") if isinstance(pred, dict) else ""
            if not prompt:
                prompt = getattr(pred, "prompt", "")
            
            if not prompt:
                return 0.0
            
            # Extract analysis and reference paths from gold example
            analysis_json = gold.get("analysis_json", "") if isinstance(gold, dict) else getattr(gold, "analysis_json", "")
            reference_paths = gold.get("reference_paths", []) if isinstance(gold, dict) else getattr(gold, "reference_paths", [])
            
            if not analysis_json or not reference_paths:
                # Fallback to keyword-based scoring if we don't have the data
                return float(score_prompt(prompt))
            
            # Parse analysis result
            from src.services.vision import AnalysisResult
            analysis_result = AnalysisResult.model_validate_json(analysis_json)
            
            # Resolve reference image paths
            abs_refs = []
            for ref in reference_paths:
                ref_path = Path(ref)
                if ref_path.is_absolute():
                    abs_refs.append(str(ref_path))
                else:
                    # Try relative to project root or assets directory
                    candidate = Path('assets') / ref
                    if candidate.exists():
                        abs_refs.append(str(candidate.resolve()))
                    else:
                        abs_refs.append(str(Path(ref).resolve()))
            
            # Get target texture path from dataset if available
            # Find matching texture in dataset
            target_texture = None
            for ex in dataset:
                refs = ex.get("references", [])
                if refs == reference_paths or refs == abs_refs:
                    target_texture = ex.get("texture")
                    break
            if target_texture:
                target_path = Path("assets") / target_texture if not Path(target_texture).is_absolute() else Path(target_texture)
                if target_path.exists():
                    target_texture = str(target_path)
                else:
                    target_texture = None
            
            # Generate image and evaluate it
            evaluator = get_evaluator()
            score, feedback = evaluator.generate_and_evaluate(
                analysis_result=analysis_result,
                reference_images=abs_refs,
                prompt_text=prompt,
                target_texture_path=target_texture
            )
            
            print(f"  Generated image score: {score:.3f} - {feedback[:100]}...", file=sys.stderr)
            
            return float(score)
            
        except Exception as e:
            print(f"Error in metric function: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fallback to keyword-based scoring on error
            try:
                prompt = pred.get("prompt", "") if isinstance(pred, dict) else getattr(pred, "prompt", "")
                if prompt:
                    return float(score_prompt(prompt))
            except:
                pass
            return 0.0
    
    return metric


def main():
    """Run GEPA optimization on prompt generation."""
    print("=== GEPA Prompt Optimization ===")
    
    # Configure DSPy
    print("Configuring DSPy...")
    reflection_lm = configure_dspy()
    
    if reflection_lm is None:
        print("Error: No LM configured. GEPA requires a language model for reflection.")
        print("Please set up authentication via:")
        print("  1. account.json in project root, OR")
        print("  2. GOOGLE_API_KEY in .env file, OR")
        print("  3. OPENAI_API_KEY in .env file")
        return
    
    # Load dataset and convert to DSPy Examples
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset_raw = load_dataset()
    print(f"Loaded {len(dataset_raw)} total examples (including mixed variants)")

    # Reduce evaluation subset to stabilize long GEPA runs
    selected_dataset = dataset_raw
    if MAX_TRAINSET_SIZE and len(dataset_raw) > MAX_TRAINSET_SIZE:
        random.seed(42)
        chosen_indices = sorted(random.sample(range(len(dataset_raw)), MAX_TRAINSET_SIZE))
        selected_dataset = [dataset_raw[i] for i in chosen_indices]
        print(f"Using {len(selected_dataset)} examples for GEPA (subset of full dataset)")
    else:
        print("Using full dataset for GEPA")
    
    # Convert to DSPy Examples
    trainset = [
        dspy.Example(
            analysis_json=json.dumps(ex["analysis"]),
            reference_paths=ex["references"],
        ).with_inputs("analysis_json", "reference_paths")
        for ex in selected_dataset
    ]
    
    # Create initial program
    print("Initializing PromptComposer...")
    program = PromptComposer()
    
    # Test initial program
    print("\nTesting initial program...")
    sample = trainset[0]
    initial_result = program(
        analysis_json=sample.analysis_json,
        reference_paths=sample.reference_paths
    )
    initial_score = score_prompt(initial_result["prompt"])
    print(f"Initial prompt score: {initial_score:.3f}")
    
    # Create metric function
    metric_fn = create_gepa_metric(selected_dataset)
    
    # Run GEPA optimization
    print("\nRunning GEPA optimization...")
    print("This may take a while...")
    
    # Create GEPA optimizer with reduced budget to avoid issues
    # GEPA requires exactly ONE of: max_metric_calls, max_full_evals, or auto
    optimizer = dspy.GEPA(
        metric=metric_fn,
        reflection_lm=reflection_lm,  # Required: LM for reflection and instruction proposal
        max_metric_calls=24,  # Slightly lower budget to reduce long blocking calls
        skip_perfect_score=False,  # Allow optimization even if initial score is high
    )
    
    try:
        optimized = optimizer.compile(
            student=program,
            trainset=trainset,
        )
        
        # Test optimized program
        print("\nTesting optimized program...")
        optimized_result = optimized(
            analysis_json=sample.analysis_json,
            reference_paths=sample.reference_paths
        )
        optimized_score = score_prompt(optimized_result["prompt"])
        print(f"Optimized prompt score: {optimized_score:.3f}")
        print(f"Improvement: {optimized_score - initial_score:+.3f}")
        
        # Save optimized prompt
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(optimized_result["prompt"], encoding='utf-8')
        print(f"\nOptimized prompt saved to {OUTPUT_PATH}")
        
        # Also save comparison
        comparison_path = OUTPUT_PATH.parent / "gepa_comparison.txt"
        comparison_path.write_text(
            f"=== INITIAL PROMPT (score: {initial_score:.3f}) ===\n\n"
            f"{initial_result['prompt']}\n\n"
            f"=== OPTIMIZED PROMPT (score: {optimized_score:.3f}) ===\n\n"
            f"{optimized_result['prompt']}\n",
            encoding='utf-8'
        )
        print(f"Comparison saved to {comparison_path}")
        
    except Exception as e:
        print(f"\nError during optimization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
