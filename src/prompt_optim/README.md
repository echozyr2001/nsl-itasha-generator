# Prompt Optimization with GEPA

This module implements prompt optimization using DSPy's GEPA (Generative Evolution for Prompt Adaptation) algorithm.

## Overview

GEPA optimizes the prompts used for image generation by:
1. **Evolutionary mutation**: Creating variations of prompts through mutations
2. **Reflection**: Using LLM to analyze execution traces and suggest improvements
3. **Pareto-based selection**: Selecting non-dominated prompt candidates

## Files

- `dspy_config.py`: Configures DSPy with language models (Gemini/OpenAI)
- `prompt_composer.py`: DSPy Module that composes prompts from analysis results
- `eval_rules.py`: Evaluation rules that score prompt quality
- `gepa_runner.py`: Main script to run GEPA optimization

## Usage

### Prerequisites

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment variables (`.env`):
   ```env
   GOOGLE_API_KEY=your_key_here
   # or
   OPENAI_API_KEY=your_key_here
   ```

3. Ensure dataset exists:
   - `datasets/gepa_dataset.json` - Training dataset
   - Reference images should be accessible

### Run Optimization

```bash
uv run python -m src.prompt_optim.gepa_runner
```

This will:
1. Load the dataset from `datasets/gepa_dataset.json`
2. Initialize the PromptComposer module
3. Run GEPA optimization for 3 rounds with up to 3 mutations per round
4. Save optimized prompt to `datasets/gepa_optimized_prompt.txt`
5. Save comparison to `datasets/gepa_comparison.txt`

### Evaluation Rules

The `eval_rules.py` module scores prompts based on:
- Mask overlay instructions
- Divider alignment specifications
- Screen avoidance guidance
- Reference image reuse instructions
- Layout slot usage
- Multi-image handling
- Coordinate precision
- Hardware avoidance
- Aspect ratio specification
- Seamless composition requirements

## Dataset Format

The dataset JSON file should contain examples like:

```json
{
  "texture": "ref/1-b.JPG",
  "analysis": {
    "images": [...],
    "synthesis": "...",
    "layout_slots": [...],
    "front_back_divider_y": 52.5
  },
  "references": ["dspy_inputs/1-b_slot0.png", ...]
}
```

## Output

After optimization, you'll get:
- `datasets/gepa_optimized_prompt.txt`: The optimized prompt text
- `datasets/gepa_comparison.txt`: Side-by-side comparison of initial vs optimized prompts

