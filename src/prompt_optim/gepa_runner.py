"""GEPA optimizer runner for prompt optimization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dspy

from src.prompt_optim.prompt_composer import PromptComposer
from src.prompt_optim.eval_rules import score_prompt
from src.prompt_optim.dspy_config import configure_dspy

DATASET_PATH = Path("datasets/gepa_dataset.json")
OUTPUT_PATH = Path("datasets/gepa_optimized_prompt.txt")


def load_dataset():
    """Load the GEPA training dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    return json.loads(DATASET_PATH.read_text())


def create_gepa_metric():
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
            # Load dataset to find target texture
            target_texture = None
            try:
                dataset = load_dataset()
                for ex in dataset:
                    if ex.get("references") == reference_paths or ex.get("references") == abs_refs:
                        target_texture = ex.get("texture")
                        if target_texture:
                            target_path = Path("assets") / target_texture if not Path(target_texture).is_absolute() else Path(target_texture)
                            if target_path.exists():
                                target_texture = str(target_path)
                            else:
                                target_texture = None
                        break
            except:
                pass  # Ignore errors in finding target texture
            
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
    print(f"Loaded {len(dataset_raw)} examples")
    
    # Convert to DSPy Examples
    trainset = [
        dspy.Example(
            analysis_json=json.dumps(ex["analysis"]),
            reference_paths=ex["references"],
        ).with_inputs("analysis_json", "reference_paths")
        for ex in dataset_raw
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
    metric_fn = create_gepa_metric()
    
    # Run GEPA optimization
    print("\nRunning GEPA optimization...")
    print("This may take a while...")
    
    # Create GEPA optimizer with reduced budget to avoid issues
    # GEPA requires exactly ONE of: max_metric_calls, max_full_evals, or auto
    optimizer = dspy.GEPA(
        metric=metric_fn,
        reflection_lm=reflection_lm,  # Required: LM for reflection and instruction proposal
        max_metric_calls=30,  # Limit metric calls (each generates an image, so limit total images)
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
