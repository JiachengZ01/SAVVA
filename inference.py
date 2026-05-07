#!/usr/bin/env python3
"""
SAVVA Inference Script for VLM Hallucination Evaluation

Supports:
- Models: llava, qwen, internvl
- Strategies: baseline (no modification), savva (our method)
- Benchmarks: amber, pope (coco/aokvqa/gqa/all), chair, shr

Usage:
    # AMBER benchmark (generative + discriminative)
    python inference.py --model internvl --strategy savva --dataset amber

    # AMBER benchmark (generative only)
    python inference.py --model internvl --strategy savva --dataset amber --amber-task generative

    # POPE benchmark (--pope-dataset: coco/aokvqa/gqa/all, --pope-type: random/popular/adversarial/all)
    python inference.py --model internvl --strategy savva --dataset pope --pope-dataset all --pope-type all

    # CHAIR benchmark
    python inference.py --model internvl --strategy savva --dataset chair

    # SHR benchmark (requires OpenAI API key for LLM-as-a-judge evaluation, default: gpt-5-mini)
    python inference.py --model internvl --strategy savva --dataset shr --shr-api-key YOUR_KEY

    # Baseline (no attention modification)
    python inference.py --model internvl --strategy baseline --dataset chair

SAVVA parameters are loaded from configs/ours.yaml (can be overridden via CLI for sensitivity analysis)
Summary results are saved to output/{dataset}/{model}/
"""

import os
import sys
import gc
import json
import argparse
import warnings
from datetime import datetime

import yaml
import torch

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*past_key_value.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import NoBoostStrategy
from strategies import SAVVAStrategy, RiskLogitsProcessor
from strategies.ours import load_savva_config, get_model_config
from evaluation.amber_eval import run_amber_evaluation, setup_seed
from evaluation.pope_eval import run_pope_evaluation_all, run_pope_evaluation, POPE_TYPES, POPE_DATASETS
from evaluation.chair_eval import run_chair_evaluation
from evaluation.shr_eval import load_shr_queries, evaluate_shr, SHR_PROMPT


# =============================================================================
# Configuration Loading
# =============================================================================

def get_savva_config(model_name):
    """Get SAVVA parameters for a specific model from configs/ours.yaml."""
    return get_model_config(model_name)


# =============================================================================
# Model Creation
# =============================================================================

def create_model(model_name, strategy_name, savva_config=None, savva_overrides=None):
    """Create model with specified strategy.

    Args:
        model_name: 'llava', 'qwen', or 'internvl'
        strategy_name: 'baseline' or 'savva'
        savva_config: SAVVA configuration dict (from get_savva_config)
        savva_overrides: Dict of SAVVA parameter overrides for sensitivity analysis

    Returns:
        (model, strategy) tuple
    """
    if strategy_name == "savva":
        if savva_config is None:
            savva_config = get_savva_config(model_name)
        # Apply any CLI overrides for sensitivity analysis
        overrides = savva_overrides or {}
        strategy = SAVVAStrategy(
            model_name=model_name,
            risk_scale=overrides.get('risk_scale'),
            m_max_visual=overrides.get('m_max_visual'),
            m_max_text=overrides.get('m_max_text'),
            vge_alpha=overrides.get('vge_alpha'),
        )
        strategy.enable()
    else:
        strategy = NoBoostStrategy()

    if model_name == "llava":
        from llava_next import LLaVABoostedInterface
        model = LLaVABoostedInterface(load_in_4bit=True, strategy=strategy)
    elif model_name == "qwen":
        from qwen3_vl import QwenBoostedInterface
        model = QwenBoostedInterface(strategy=strategy)
    elif model_name == "internvl":
        from internvl3_5 import InternVLBoostedInterface
        model = InternVLBoostedInterface(strategy=strategy)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, strategy


def create_savva_processor(strategy, debug=False):
    """Factory function to create SAVVA logits processor."""
    return RiskLogitsProcessor(strategy, debug=debug)


# =============================================================================
# Dataset Runners
# =============================================================================

def run_amber(model, model_name, strategy, strategy_name, args, savva_processor_factory=None):
    """Run AMBER benchmark evaluation."""
    result = run_amber_evaluation(
        model=model,
        model_name=model_name,
        strategy=strategy,
        strategy_name=strategy_name,
        amber_dir=args.data_dir or "datasets/amber",
        num_gen_samples=args.num_samples,
        num_disc_samples=args.num_samples,
        savva_processor_factory=savva_processor_factory,
        task_type=args.amber_task,
    )

    return {
        'dataset': 'amber',
        'amber_task': args.amber_task,
        'metrics': {
            'generative': result['generative_metrics'],
            'discriminative': result['discriminative_metrics'],
            'amber_score': result['amber_score'],
        },
        'responses': {
            'generative': result['generative_results'],
            'discriminative': result['discriminative_results'],
        }
    }


def run_pope(model, model_name, strategy, strategy_name, args, savva_processor_factory=None):
    """Run POPE benchmark evaluation."""
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pope_dir = args.data_dir or os.path.join(script_dir, "datasets", "pope")

    # Image directory paths for each dataset
    image_dir_map = {
        "coco": os.path.join(script_dir, "datasets", "COCO_2014", "val2014"),
        "aokvqa": os.path.join(script_dir, "datasets", "COCO_2014", "val2014"),  # A-OKVQA uses COCO images
        "gqa": os.path.join(script_dir, "datasets", "gqa", "images"),  # GQA has its own images
    }

    # Parse pope_datasets: "all" -> all datasets, or specific datasets
    if args.pope_dataset == "all":
        pope_datasets = POPE_DATASETS  # ["coco", "aokvqa", "gqa"]
    else:
        pope_datasets = [args.pope_dataset]

    # Run evaluation for each dataset
    all_metrics = {}  # {dataset: {pope_type: metrics}}
    all_responses = {}  # {dataset: {pope_type: responses}}

    for pope_dataset in pope_datasets:
        image_dir = image_dir_map[pope_dataset]

        if args.pope_type == "all":
            result = run_pope_evaluation_all(
                model=model,
                model_name=model_name,
                strategy=strategy,
                strategy_name=strategy_name,
                pope_dir=pope_dir,
                image_dir=image_dir,
                num_samples=args.num_samples,
                savva_processor_factory=savva_processor_factory,
                dataset=pope_dataset,
            )
            all_metrics[pope_dataset] = {pope_type: result[pope_type]['metrics'] for pope_type in POPE_TYPES}
            all_responses[pope_dataset] = {pope_type: result[pope_type]['results'] for pope_type in POPE_TYPES}
        else:
            result = run_pope_evaluation(
                model=model,
                model_name=model_name,
                strategy=strategy,
                strategy_name=strategy_name,
                pope_dir=pope_dir,
                image_dir=image_dir,
                pope_type=args.pope_type,
                num_samples=args.num_samples,
                savva_processor_factory=savva_processor_factory,
                dataset=pope_dataset,
            )
            all_metrics[pope_dataset] = {args.pope_type: result['metrics']}
            all_responses[pope_dataset] = {args.pope_type: result['results']}

    return {
        'dataset': 'pope',
        'pope_datasets': pope_datasets,
        'pope_type': args.pope_type,
        'metrics': all_metrics,
        'responses': all_responses,
    }


def run_chair(model, model_name, strategy, strategy_name, args, savva_processor_factory=None):
    """Run CHAIR benchmark evaluation."""
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coco_dir = args.data_dir or os.path.join(script_dir, "datasets", "COCO_2014")
    cache_path = os.path.join(script_dir, "datasets", "chair", "chair.pkl")

    result = run_chair_evaluation(
        model=model,
        model_name=model_name,
        strategy=strategy,
        strategy_name=strategy_name,
        coco_dir=coco_dir,
        cache_path=cache_path,
        num_samples=args.num_samples,
        savva_processor_factory=savva_processor_factory,
    )

    return {
        'dataset': 'chair',
        'metrics': result['metrics'],
        'responses': result['results'],
    }


def run_shr(model, model_name, strategy, strategy_name, args, savva_processor_factory=None):
    """Run SHR benchmark evaluation."""
    from PIL import Image
    from transformers import LogitsProcessorList
    from tqdm import tqdm

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shr_dir = os.path.join(script_dir, "datasets", "shr")
    vg_dir = os.path.join(script_dir, "datasets", "VG")

    # Check if using pre-generated captions (skip inference)
    if args.shr_captions_file:
        print(f"Loading pre-generated captions from: {args.shr_captions_file}")
        with open(args.shr_captions_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} captions, skipping inference.")
    else:
        # Load queries
        queries = load_shr_queries(shr_dir, vg_dir, num_samples=args.num_samples)

        # Setup processor
        savva_processor = savva_processor_factory(strategy) if savva_processor_factory else None

        # Generate captions
        results = {}
        for item in tqdm(queries, desc="SHR Inference"):
            image_path = item["image_path"]
            image_id = item["image_id"]

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_id}: {e}")
                continue

            # Prepare inputs
            inputs = model.prepare_inputs(image, SHR_PROMPT)
            if inputs is None:
                continue

            # Set visual indices if strategy is used
            if strategy is not None and hasattr(strategy, 'reset'):
                visual_indices = model.detect_visual_token_indices(inputs['input_ids'])
                model._visual_indices_tensor = torch.tensor(
                    visual_indices, dtype=torch.long, device=model.device
                )
                strategy.reset()
                if savva_processor:
                    savva_processor.step_count = 0

                # Setup VGE visual grounding if needed
                if hasattr(strategy, 'set_grounding_scores'):
                    if hasattr(model, 'setup_vge_grounding'):
                        model.setup_vge_grounding(inputs, strategy)

            # Get pad_token_id
            if hasattr(model.processor, 'tokenizer') and model.processor.tokenizer.pad_token_id is not None:
                pad_token_id = model.processor.tokenizer.pad_token_id
            elif hasattr(model.model.config, 'pad_token_id') and model.model.config.pad_token_id is not None:
                pad_token_id = model.model.config.pad_token_id
            else:
                pad_token_id = model.model.config.eos_token_id

            # Generate (greedy decoding for fair comparison across all methods)
            # Note: Greedy decoding for fair comparison
            gen_kwargs = {
                **inputs,
                "max_new_tokens": 512,
                "do_sample": False,
                "num_beams": 1,  # Greedy decoding for fair comparison
                "pad_token_id": pad_token_id,
            }

            # Add logits processor
            if savva_processor:
                gen_kwargs["logits_processor"] = LogitsProcessorList([savva_processor])

            with torch.no_grad():
                outputs = model.model.generate(**gen_kwargs)

            # Decode
            if hasattr(model.processor, 'tokenizer'):
                caption = model.processor.tokenizer.decode(
                    outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )
            else:
                caption = model.processor.decode(
                    outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )

            results[str(image_id)] = caption

        # Save captions to file (so we can re-run judge without re-running inference)
        captions_file = os.path.join(args.output_dir, "shr_captions.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(captions_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCaptions saved to: {captions_file}")

    # Run GPT evaluation if API key is provided
    metrics = None
    judgement = None
    if args.shr_api_key:
        eval_result = evaluate_shr(
            json_file=results,
            shr_path=shr_dir,
            vg_path=vg_dir,
            api_key=args.shr_api_key,
            base_url=args.shr_base_url,
            model_name=args.shr_model_name,
            no_gpt_judge=False,
        )
        metrics = eval_result['metrics']
        judgement = eval_result['judgement']
    else:
        print("\nNote: --shr-api-key not provided. Skipping LLM-as-a-judge evaluation.")
        print("To run full evaluation, provide OpenAI API key with --shr-api-key")

    return {
        'dataset': 'shr',
        'metrics': metrics,
        'responses': results,
        'judgement': judgement,
    }


# Dataset registry
DATASET_RUNNERS = {
    'amber': run_amber,
    'pope': run_pope,
    'chair': run_chair,
    'shr': run_shr,
}


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Inference Script for VLM Hallucination Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py --model internvl --strategy savva --dataset amber
    python inference.py --model internvl --strategy savva --dataset pope --pope-dataset all
    python inference.py --model internvl --strategy savva --dataset chair
    python inference.py --model internvl --strategy savva --dataset shr --shr-api-key YOUR_KEY

    # SAVVA with parameter overrides (sensitivity analysis):
    python inference.py --model internvl --strategy savva --dataset chair --vge-alpha 0.7
    python inference.py --model internvl --strategy savva --dataset amber --risk-scale 0.5 --m-max-visual 1.3

    # Baseline (no attention modification):
    python inference.py --model internvl --strategy baseline --dataset chair
        """
    )

    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        choices=["llava", "qwen", "internvl"],
                        help="Model to test")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["baseline", "savva"],
                        help="Strategy to use (baseline=no modification, savva=our method)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_RUNNERS.keys()),
                        help="Dataset to evaluate on")

    # Optional arguments
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset directory (default: datasets/<dataset>)")
    parser.add_argument("--pope-type", type=str, default="all",
                        choices=["random", "popular", "adversarial", "all"],
                        help="POPE evaluation type (default: all)")
    parser.add_argument("--pope-dataset", type=str, default="coco",
                        choices=POPE_DATASETS + ["all"],
                        help="POPE dataset: coco, aokvqa, gqa, or all (default: coco)")
    parser.add_argument("--amber-task", type=str, default="all",
                        choices=["generative", "discriminative", "all"],
                        help="AMBER task type: generative, discriminative, or all (default: all)")
    parser.add_argument("--shr-api-key", type=str, default=None,
                        help="OpenAI API key for SHR LLM-as-a-judge evaluation")
    parser.add_argument("--shr-base-url", type=str, default=None,
                        help="OpenAI API base URL for SHR evaluation (optional)")
    parser.add_argument("--shr-model-name", type=str, default="gpt-5-mini",
                        help="LLM judge model name for SHR evaluation (default: gpt-5-mini)")
    parser.add_argument("--shr-captions-file", type=str, default=None,
                        help="Path to pre-generated captions JSON file (skip inference, run judge only)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save detailed results to JSON file")

    # SAVVA parameter overrides (for sensitivity analysis)
    parser.add_argument("--risk-scale", type=float, default=None,
                        help="SAVVA: Risk scaling factor")
    parser.add_argument("--m-max-visual", type=float, default=None,
                        help="SAVVA: Maximum visual token boost multiplier")
    parser.add_argument("--m-max-text", type=float, default=None,
                        help="SAVVA: Maximum text token suppress multiplier (suppress = 1/m_max_text)")
    parser.add_argument("--vge-alpha", type=float, default=None,
                        help="SAVVA: VGE weight for entropy (alpha*entropy + (1-alpha)*(1-G))")

    return parser.parse_args()


def print_summary(result, model_name, strategy_name, dataset,
                  save_path=None, num_samples=None, output_dir="output"):
    """Print evaluation summary and optionally save to file.

    Args:
        result: Evaluation result dict
        model_name: Model name
        strategy_name: Strategy name
        dataset: Dataset name (amber, pope, chair, shr)
        save_path: If provided, save summary to this path. If True, auto-generate path.
        num_samples: Number of samples evaluated (None = all)
        output_dir: Output directory for auto-generated path (default: output)
    """
    import numpy as np

    metrics = result['metrics']
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    dataset_str = dataset
    if dataset == 'pope' and 'pope_datasets' in result:
        dataset_str = f"pope-{'+'.join(result['pope_datasets'])}"
    lines.append(f"Model: {model_name}, Strategy: {strategy_name}, Dataset: {dataset_str}")
    lines.append("-" * 70)

    # Generate filename for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_str = f"n{num_samples}" if num_samples else "all"
    filename = None

    if dataset == 'amber':
        gen = metrics['generative']
        disc = metrics['discriminative']
        amber_task = result.get('amber_task', 'all')
        filename = f"{strategy_name}_{amber_task}_{sample_str}_{timestamp}.txt"

        lines.append(f"{'Metric':<20} {'Value':>10}")
        lines.append("-" * 70)

        if gen:
            lines.append(f"{'CHAIR ↓':<20} {gen['CHAIR']:>9.2f}%")
            lines.append(f"{'Cover ↑':<20} {gen['Cover']:>9.2f}%")
            lines.append(f"{'Hal ↓':<20} {gen['Hal']:>9.2f}%")
            lines.append(f"{'Cog ↓':<20} {gen['Cog']:>9.2f}%")

        if gen and disc:
            lines.append("-" * 70)

        if disc:
            lines.append(f"{'Accuracy ↑':<20} {disc['Accuracy']:>9.2f}%")
            lines.append(f"{'Precision ↑':<20} {disc['Precision']:>9.2f}%")
            lines.append(f"{'Recall ↑':<20} {disc['Recall']:>9.2f}%")
            lines.append(f"{'F1 ↑':<20} {disc['F1']:>9.2f}%")

        if metrics['amber_score'] is not None:
            lines.append("-" * 70)
            lines.append(f"{'AMBER Score ↑':<20} {metrics['amber_score']:>9.2f}%")
            lines.append("=" * 70)
            lines.append("")
            lines.append("AMBER Score = (1 - CHAIR + F1) / 2")
        else:
            lines.append("=" * 70)
            if amber_task == "generative":
                lines.append("")
                lines.append("(Only generative task was run)")
            elif amber_task == "discriminative":
                lines.append("")
                lines.append("(Only discriminative task was run)")

    elif dataset == 'pope':
        pope_datasets = result.get('pope_datasets', ['coco'])
        pope_type = result.get('pope_type', 'all')
        pope_str = '+'.join(pope_datasets)
        filename = f"{pope_str}_{strategy_name}_{pope_type}_{sample_str}_{timestamp}.txt"

        # Collect all metrics across datasets for averaging
        all_acc, all_prec, all_recall, all_f1 = [], [], [], []

        for pope_dataset in pope_datasets:
            dataset_metrics = metrics[pope_dataset]

            lines.append(f"")
            lines.append(f"[{pope_dataset.upper()}]")
            lines.append(f"{'Type':<12} {'Acc ↑':>8} {'Prec ↑':>8} {'Recall ↑':>8} {'F1 ↑':>8} {'Yes%':>8}")
            lines.append("-" * 60)

            dataset_acc, dataset_prec, dataset_recall, dataset_f1 = [], [], [], []
            for pope_type in ['random', 'popular', 'adversarial']:
                if pope_type in dataset_metrics:
                    m = dataset_metrics[pope_type]
                    lines.append(f"{pope_type:<12} {m['Accuracy']:>7.2f}% {m['Precision']:>7.2f}% "
                                 f"{m['Recall']:>7.2f}% {m['F1']:>7.2f}% {m['Yes_ratio']:>7.2f}%")
                    dataset_acc.append(m['Accuracy'])
                    dataset_prec.append(m['Precision'])
                    dataset_recall.append(m['Recall'])
                    dataset_f1.append(m['F1'])
                    all_acc.append(m['Accuracy'])
                    all_prec.append(m['Precision'])
                    all_recall.append(m['Recall'])
                    all_f1.append(m['F1'])

            if len(dataset_acc) > 1:
                lines.append("-" * 60)
                lines.append(f"{'Avg':<12} {np.mean(dataset_acc):>7.2f}% {np.mean(dataset_prec):>7.2f}% "
                             f"{np.mean(dataset_recall):>7.2f}% {np.mean(dataset_f1):>7.2f}%")

        # Print overall average if multiple datasets
        if len(pope_datasets) > 1:
            lines.append("")
            lines.append("=" * 70)
            lines.append("OVERALL AVERAGE (across all datasets)")
            lines.append("=" * 70)
            lines.append(f"Accuracy:  {np.mean(all_acc):.2f}%")
            lines.append(f"Precision: {np.mean(all_prec):.2f}%")
            lines.append(f"Recall:    {np.mean(all_recall):.2f}%")
            lines.append(f"F1:        {np.mean(all_f1):.2f}%")

        lines.append("=" * 70)

    elif dataset == 'chair':
        filename = f"{strategy_name}_{sample_str}_{timestamp}.txt"

        lines.append(f"{'Metric':<20} {'Value':>10}")
        lines.append("-" * 70)
        lines.append(f"{'CHAIRs ↓':<20} {metrics['CHAIRs']:>9.2f}%")
        lines.append(f"{'CHAIRi ↓':<20} {metrics['CHAIRi']:>9.2f}%")
        lines.append(f"{'Recall ↑':<20} {metrics['Recall']:>9.2f}%")
        lines.append(f"{'Precision ↑':<20} {metrics['Precision']:>9.2f}%")
        lines.append(f"{'F1 ↑':<20} {metrics['F1']:>9.2f}%")
        lines.append(f"{'Avg Length':<20} {metrics['Len']:>9.1f}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("CHAIRs = % of captions with hallucination")
        lines.append("CHAIRi = % of mentioned objects that are hallucinated")

    elif dataset == 'shr':
        filename = f"{strategy_name}_{sample_str}_{timestamp}.txt"

        if metrics is None:
            lines.append("Note: LLM-as-a-judge evaluation was not run (no API key provided)")
            lines.append("Captions have been generated. Run with --shr-api-key for full evaluation.")
        else:
            lines.append(f"{'Metric':<25} {'Value':>10}")
            lines.append("-" * 70)
            lines.append(f"{'HSR (hal_sents_ratio) ↓':<25} {metrics['run1']['hal_sents_ratio']*100:>9.2f}%")
            lines.append(f"{'HWR (hal_words_ratio) ↓':<25} {metrics['run1']['hal_words_ratio']*100:>9.2f}%")
            lines.append(f"{'SPI (sents_per_image) ↑':<25} {metrics['run1']['sents_per_image']:>9.2f}")
            lines.append(f"{'WPI (words_per_image) ↑':<25} {metrics['run1']['words_per_image']:>9.2f}")
            lines.append(f"{'HSPI (hal_sents_per_img)':<25} {metrics['run1']['hal_sents_per_image']:>9.2f}")
            lines.append(f"{'HWPI (hal_words_per_img)':<25} {metrics['run1']['hal_words_per_image']:>9.2f}")
            lines.append("-" * 70)
            lines.append(f"{'gram-1 repetition':<25} {metrics['gram-1-repetition']:>9.3f}")
            lines.append(f"{'gram-2 repetition':<25} {metrics['gram-2-repetition']:>9.3f}")
            lines.append(f"{'gram-3 repetition':<25} {metrics['gram-3-repetition']:>9.3f}")
            lines.append(f"{'gram-4 repetition':<25} {metrics['gram-4-repetition']:>9.3f}")
            lines.append("=" * 70)
            lines.append("")
            lines.append("HSR = Hallucinated Sentences Ratio (main metric)")
            lines.append("HWR = Hallucinated Words Ratio")

    # Print to console
    for line in lines:
        print(line)

    # Save to file if requested
    if save_path:
        if save_path is True:
            # Auto-generate path
            save_dir = os.path.join(output_dir, dataset, model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nSummary saved to: {save_path}")
        return save_path

    return None


def main():
    args = parse_args()
    setup_seed(args.seed)

    # Load strategy config if needed
    savva_config = None
    if args.strategy == "savva":
        savva_config = get_savva_config(args.model)

    # Print configuration
    print("=" * 70)
    print("VLM Hallucination Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "amber":
        print(f"AMBER Task: {args.amber_task}")
    if args.dataset == "pope":
        if args.pope_dataset == "all":
            print(f"POPE Datasets: coco, aokvqa, gqa (all)")
        else:
            print(f"POPE Dataset: {args.pope_dataset}")
        print(f"POPE Type: {args.pope_type}")
    if args.dataset == "shr":
        print(f"SHR API Key: {'provided' if args.shr_api_key else 'not provided (will skip GPT-4 evaluation)'}")
        if args.shr_base_url:
            print(f"SHR API URL: {args.shr_base_url}")
    if savva_config:
        # Show config values with override indicators
        def fmt(val, override, name):
            if override is not None:
                return f"{override} (override)"
            return f"{val}"

        risk_str = fmt(savva_config['risk_scale'], args.risk_scale, 'risk')
        vge_alpha_str = fmt(savva_config.get('vge_alpha', 0.75), args.vge_alpha, 'vge_alpha')
        m_max_visual_str = fmt(savva_config['m_max_visual'], args.m_max_visual, 'm_max_visual')
        m_max_text_str = fmt(savva_config['m_max_text'], args.m_max_text, 'm_max_text')
        print(f"VGE alpha: {vge_alpha_str}")
        print(f"Risk scale: {risk_str}")
        print(f"M visual: [1.0, {m_max_visual_str}]")
        print(f"M text: {m_max_text_str} (suppress = 1/m_text)")
        print(f"Layers: {savva_config['start_layer']}-{savva_config['end_layer']}")
    print(f"Samples: {args.num_samples or 'all'}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Collect SAVVA overrides from CLI args (for sensitivity analysis)
    savva_overrides = None
    if args.strategy == "savva":
        savva_overrides = {}
        if args.risk_scale is not None:
            savva_overrides['risk_scale'] = args.risk_scale
        if args.m_max_visual is not None:
            savva_overrides['m_max_visual'] = args.m_max_visual
        if args.m_max_text is not None:
            savva_overrides['m_max_text'] = args.m_max_text
        if args.vge_alpha is not None:
            savva_overrides['vge_alpha'] = args.vge_alpha

    # Special case: SHR with pre-generated captions (skip model loading)
    if args.dataset == 'shr' and args.shr_captions_file:
        model = None  # No model needed for judge-only mode
        print(f"Running SHR judge only (no model loading)")
        print(f"Loading captions from: {args.shr_captions_file}")

        with open(args.shr_captions_file, 'r') as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} captions")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        shr_dir = os.path.join(script_dir, "datasets", "shr")
        vg_dir = os.path.join(script_dir, "datasets", "VG")

        if args.shr_api_key:
            eval_result = evaluate_shr(
                json_file=captions,
                shr_path=shr_dir,
                vg_path=vg_dir,
                api_key=args.shr_api_key,
                base_url=args.shr_base_url,
                model_name=args.shr_model_name,
                no_gpt_judge=False,
            )
            result = {
                'dataset': 'shr',
                'metrics': eval_result['metrics'],
                'responses': captions,
                'judgement': eval_result['judgement'],
            }
        else:
            print("Error: --shr-api-key required for judge")
            return

        # Print and save summary
        print_summary(result, args.model, args.strategy, args.dataset,
                      save_path=True, num_samples=args.num_samples, output_dir=args.output_dir)
    else:
        # Normal flow: load model and run evaluation
        # Create model
        model, strategy = create_model(args.model, args.strategy, savva_config, savva_overrides)

        # Create SAVVA processor factory if needed
        savva_processor_factory = create_savva_processor if args.strategy == "savva" else None

        # Run evaluation
        runner = DATASET_RUNNERS[args.dataset]
        result = runner(model, args.model, strategy, args.strategy, args, savva_processor_factory)

        # Print and save summary
        print_summary(result, args.model, args.strategy, args.dataset,
                      save_path=True, num_samples=args.num_samples, output_dir=args.output_dir)

    # Save results if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include pope_dataset in filename if applicable
        if args.dataset == "pope":
            result_file = os.path.join(
                args.output_dir,
                f"pope_{args.pope_dataset}_{args.model}_{args.strategy}_{timestamp}.json"
            )
        else:
            result_file = os.path.join(
                args.output_dir,
                f"{args.dataset}_{args.model}_{args.strategy}_{timestamp}.json"
            )

        save_data = {
            "config": {
                "model": args.model,
                "strategy": args.strategy,
                "dataset": args.dataset,
                "pope_dataset": args.pope_dataset if args.dataset == "pope" else None,
                "savva_config": savva_config if savva_config else None,
                "savva_overrides": savva_overrides if savva_overrides else None,
                "seed": args.seed,
            },
            **result
        }

        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    # Cleanup
    if model is not None:
        del model
        torch.cuda.empty_cache()
    gc.collect()

    return result


if __name__ == "__main__":
    main()
