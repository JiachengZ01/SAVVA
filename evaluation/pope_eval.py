#!/usr/bin/env python3
"""
POPE Evaluation Utilities

Complete evaluation pipeline for POPE (Polling-based Object Probing Evaluation) benchmark:
- Data loading (random, popular, adversarial)
- Inference
- Metrics computation (Accuracy, Precision, Recall, F1, Yes ratio)

The evaluation logic follows POPE official implementation exactly.
Reference: https://github.com/AoiDragon/POPE/blob/main/evaluate.py

Author: AdaVBoost Project
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import set_seed, LogitsProcessorList


# =============================================================================
# Constants
# =============================================================================

# POPE types
POPE_TYPES = ["random", "popular", "adversarial"]

# Supported datasets
POPE_DATASETS = ["coco", "aokvqa", "gqa"]

# Dataset file patterns
# COCO: JSONL format (one JSON per line, 6 questions per image)
# A-OKVQA/GQA: JSON array format (one question per object)
# Directory structure: datasets/pope/{dataset}/{filename}
POPE_FILE_PATTERNS = {
    "coco": "coco/coco_pope_chat_{pope_type}.json",      # JSONL format
    "aokvqa": "aokvqa/aokvqa_pope_seem_{pope_type}.json",  # JSON array format
    "gqa": "gqa/gqa_pope_seem_{pope_type}.json",        # JSON array format
}


# =============================================================================
# Utility Functions
# =============================================================================

def setup_seed(seed=42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_pope_answer(text):
    """Parse model response to yes/no following POPE official logic.

    This follows exactly the POPE evaluate.py logic:
    1. Only keep the first sentence (split by '.')
    2. Remove commas
    3. Split by space into words
    4. If 'No', 'not', or 'no' in words -> 'no'
    5. Otherwise -> 'yes'

    Args:
        text: Model response string

    Returns:
        'yes' or 'no'
    """
    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]

    # Remove commas
    text = text.replace(',', '')

    # Split by space
    words = text.split(' ')

    # Check for negative words
    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    else:
        return 'yes'


# =============================================================================
# Data Loading
# =============================================================================

def load_pope_data(pope_dir, pope_type="random", dataset="coco"):
    """Load POPE evaluation data.

    Args:
        pope_dir: Path to pope directory (e.g., datasets/pope_coco or datasets/pope_aokvqa)
        pope_type: One of "random", "popular", "adversarial"
        dataset: One of "coco", "aokvqa", "gqa"

    Returns:
        List of dicts, each containing:
        - image: Image filename (e.g., "COCO_val2014_000000310196.jpg")
        - questions: List of questions (6 for COCO, 1 for A-OKVQA/GQA)
        - labels: List of labels ("yes" or "no")
        - chat_id or question_id: ID
    """
    if pope_type not in POPE_TYPES:
        raise ValueError(f"pope_type must be one of {POPE_TYPES}, got {pope_type}")
    if dataset not in POPE_DATASETS:
        raise ValueError(f"dataset must be one of {POPE_DATASETS}, got {dataset}")

    # Get file pattern for this dataset
    file_pattern = POPE_FILE_PATTERNS[dataset].format(pope_type=pope_type)
    pope_file = os.path.join(pope_dir, file_pattern)

    data = []

    if dataset == "coco":
        # COCO format: JSONL with 6 questions per image
        with open(pope_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    'image': item['image'],
                    'questions': item['text'],  # List of 6 questions
                    'labels': item['label'],    # List of 6 labels
                    'chat_id': item['chat_id'],
                })
    else:
        # A-OKVQA/GQA format: JSON array with 1 question per object
        with open(pope_file, 'r') as f:
            items = json.load(f)

        # Group by image for consistency with COCO format
        from collections import defaultdict
        image_groups = defaultdict(lambda: {'questions': [], 'labels': [], 'question_ids': []})

        for item in items:
            img = item['image']
            image_groups[img]['questions'].append(item['text'])
            image_groups[img]['labels'].append(item['label'])
            image_groups[img]['question_ids'].append(item['question_id'])

        for img, group in image_groups.items():
            data.append({
                'image': img,
                'questions': group['questions'],
                'labels': group['labels'],
                'question_ids': group['question_ids'],
            })

    return data


def load_pope_queries(pope_dir, pope_type="random", num_samples=None, dataset="coco"):
    """Load POPE queries in flat format (one question per item).

    Args:
        pope_dir: Path to pope directory
        pope_type: One of "random", "popular", "adversarial"
        num_samples: Number of samples to load (None = all)
        dataset: One of "coco", "aokvqa", "gqa"

    Returns:
        List of dicts, each containing:
        - image: Image filename
        - question: Single question
        - label: Single label ("yes" or "no")
        - chat_id or question_id: ID
        - question_idx: Question index within the group
    """
    data = load_pope_data(pope_dir, pope_type, dataset)

    queries = []
    for item in data:
        for q_idx, (question, label) in enumerate(zip(item['questions'], item['labels'])):
            query = {
                'image': item['image'],
                'question': question,
                'label': label,
                'question_idx': q_idx,
            }
            # Add ID field (different name for different datasets)
            if 'chat_id' in item:
                query['chat_id'] = item['chat_id']
            if 'question_ids' in item:
                query['question_id'] = item['question_ids'][q_idx]
            queries.append(query)

    if num_samples is not None:
        queries = queries[:num_samples]

    return queries


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_pope_results(predictions, labels):
    """Evaluate POPE results following official POPE evaluation logic exactly.

    This follows exactly the POPE evaluate.py implementation.

    Args:
        predictions: List of model predictions ('yes' or 'no')
        labels: List of ground truth labels ('yes' or 'no')

    Returns:
        Dictionary with metrics:
        - TP, FP, TN, FN: Confusion matrix values
        - Accuracy: (TP + TN) / total
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1: 2 * P * R / (P + R)
        - Yes_ratio: Proportion of 'yes' predictions
    """
    assert len(predictions) == len(labels), \
        f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels"

    # Convert to numeric (exactly as POPE does)
    # yes -> 1, no -> 0
    pred_list = []
    for pred in predictions:
        if pred == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    label_list = []
    for label in labels:
        if label == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

    # Constants (matching POPE exactly)
    pos = 1  # yes
    neg = 0  # no

    # Calculate yes ratio
    yes_ratio = pred_list.count(1) / len(pred_list)

    # Calculate confusion matrix
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    # Calculate metrics (matching POPE exactly)
    # Note: POPE doesn't handle division by zero, we add small epsilon for safety
    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN)

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Accuracy': round(acc * 100, 2),
        'Precision': round(precision * 100, 2),
        'Recall': round(recall * 100, 2),
        'F1': round(f1 * 100, 2),
        'Yes_ratio': round(yes_ratio * 100, 2),
    }


# =============================================================================
# Inference Functions
# =============================================================================

def run_pope_inference(model, model_name, strategy, strategy_name, queries, image_dir,
                       adavboost_processor_factory=None):
    """Run POPE inference.

    Args:
        model: Model instance (LLaVABoostedInterface or QwenBoostedInterface)
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'adavboost'
        queries: List of query items from load_pope_queries()
        image_dir: Path to COCO val2014 image directory
        adavboost_processor_factory: Factory function to create AdaVBoost logits processor

    Returns:
        List of results with predictions and labels
    """
    results = []
    skipped_images = []

    # Max pixels limit for Qwen to prevent OOM
    QWEN_MAX_PIXELS = 1280 * 28 * 28  # ~1M pixels

    for item in tqdm(queries, desc=f"{model_name} POPE {strategy_name}"):
        img_path = os.path.join(image_dir, item["image"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Skip if image can't be loaded
            results.append({
                "image": item["image"],
                "question": item["question"],
                "label": item["label"],
                "response": "",
                "prediction": None,  # None indicates skipped
                "skipped": True,
            })
            continue

        # Skip oversized images to prevent OOM
        if QWEN_MAX_PIXELS is not None:
            w, h = image.size
            if w * h > QWEN_MAX_PIXELS:
                skipped_images.append({
                    "image": item["image"],
                    "size": f"{w}x{h}",
                    "pixels": w * h
                })
                results.append({
                    "image": item["image"],
                    "question": item["question"],
                    "label": item["label"],
                    "response": "",
                    "prediction": None,  # None indicates skipped
                    "skipped": True,
                })
                continue

        # Prepare inputs
        inputs = model.prepare_inputs(image, item["question"])
        if inputs is None:
            results.append({
                "image": item["image"],
                "question": item["question"],
                "label": item["label"],
                "response": "",
                "prediction": None,  # None indicates skipped
                "skipped": True,
            })
            continue

        visual_indices = model.detect_visual_token_indices(inputs['input_ids'])

        # Set visual indices and reset strategy for AdaVBoost
        if strategy_name == "adavboost":
            model._visual_indices_tensor = torch.tensor(
                visual_indices, dtype=torch.long, device=model.device
            )
            strategy.reset()
            # Setup VGE visual grounding if needed
            if hasattr(strategy, 'set_grounding_scores'):
                if hasattr(model, 'setup_vge_grounding'):
                    model.setup_vge_grounding(inputs, strategy)

        # Get pad_token_id to avoid warning
        if hasattr(model.processor, 'tokenizer') and model.processor.tokenizer.pad_token_id is not None:
            pad_token_id = model.processor.tokenizer.pad_token_id
        elif hasattr(model.model.config, 'pad_token_id') and model.model.config.pad_token_id is not None:
            pad_token_id = model.model.config.pad_token_id
        else:
            pad_token_id = model.model.config.eos_token_id

        gen_kwargs = {
            **inputs,
            "max_new_tokens": 64,  # Short answer expected
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }

        # Add AdaVBoost logits processor if needed
        if strategy_name == "adavboost" and adavboost_processor_factory:
            adavboost_processor = adavboost_processor_factory(strategy)
            gen_kwargs["logits_processor"] = LogitsProcessorList([adavboost_processor])

        with torch.no_grad():
            outputs = model.model.generate(**gen_kwargs)

        # Decode response
        if model_name == "llava":
            response = model.processor.tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )
        else:
            response = model.processor.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )

        # Parse answer following POPE official logic
        prediction = parse_pope_answer(response)

        results.append({
            "image": item["image"],
            "question": item["question"],
            "label": item["label"],
            "response": response,
            "prediction": prediction,
        })

    # Report skipped images
    if skipped_images:
        print(f"\n  Skipped {len(skipped_images)} oversized images (>{QWEN_MAX_PIXELS:,} pixels)")

    return results


def run_pope_evaluation(model, model_name, strategy, strategy_name,
                        pope_dir, image_dir, pope_type="random",
                        num_samples=None, adavboost_processor_factory=None, dataset="coco"):
    """Run complete POPE evaluation for a single pope_type.

    Args:
        model: Model instance
        model_name: 'llava', 'qwen', or 'internvl'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'adavboost'
        pope_dir: Path to pope directory (e.g., datasets/pope_coco)
        image_dir: Path to image directory (e.g., datasets/COCO_2014/val2014)
        pope_type: One of "random", "popular", "adversarial"
        num_samples: Number of samples (None = all)
        adavboost_processor_factory: Factory function to create AdaVBoost logits processor
        dataset: One of "coco", "aokvqa", "gqa"

    Returns:
        Dictionary with metrics and raw results
    """
    queries = load_pope_queries(pope_dir, pope_type, num_samples, dataset)

    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} POPE-{dataset.upper()}-{pope_type} - {strategy_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Loaded {len(queries)} questions from {dataset}")

    # Run inference
    results = run_pope_inference(
        model, model_name, strategy, strategy_name,
        queries, image_dir, adavboost_processor_factory
    )

    # Extract predictions and labels (filter out skipped)
    valid_results = [r for r in results if not r.get('skipped', False)]
    skipped_count = len(results) - len(valid_results)

    predictions = [r['prediction'] for r in valid_results]
    labels = [r['label'] for r in valid_results]

    # Evaluate
    metrics = evaluate_pope_results(predictions, labels)

    if skipped_count > 0:
        print(f"\n  Skipped {skipped_count} samples (not included in metrics)")

    print(f"\nResults (n={len(valid_results)}):")
    print(f"  TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
    print(f"  Accuracy:  {metrics['Accuracy']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.2f}%")
    print(f"  Recall:    {metrics['Recall']:.2f}%")
    print(f"  F1:        {metrics['F1']:.2f}%")
    print(f"  Yes_ratio: {metrics['Yes_ratio']:.2f}%")

    return {
        'metrics': metrics,
        'results': results,
        'pope_type': pope_type,
        'dataset': dataset,
    }


def run_pope_evaluation_all(model, model_name, strategy, strategy_name,
                            pope_dir, image_dir, num_samples=None,
                            adavboost_processor_factory=None, dataset="coco"):
    """Run POPE evaluation on all three types (random, popular, adversarial).

    Args:
        model: Model instance
        model_name: 'llava', 'qwen', or 'internvl'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'adavboost'
        pope_dir: Path to pope directory
        image_dir: Path to image directory
        num_samples: Number of samples per type (None = all)
        adavboost_processor_factory: Factory function to create AdaVBoost logits processor
        dataset: One of "coco", "aokvqa", "gqa"

    Returns:
        Dictionary with results for all three types
    """
    all_results = {}

    for pope_type in POPE_TYPES:
        result = run_pope_evaluation(
            model, model_name, strategy, strategy_name,
            pope_dir, image_dir, pope_type,
            num_samples, adavboost_processor_factory, dataset
        )
        all_results[pope_type] = result

    return all_results


# =============================================================================
# Standalone Evaluation (for evaluating saved results)
# =============================================================================

def evaluate_saved_results(results_file, pope_dir, pope_type, dataset="coco"):
    """Evaluate saved inference results.

    Args:
        results_file: Path to JSON file with results
        pope_dir: Path to pope directory (for ground truth)
        pope_type: One of "random", "popular", "adversarial"
        dataset: One of "coco", "aokvqa", "gqa"

    Returns:
        Metrics dictionary
    """
    # Load predictions
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load ground truth
    queries = load_pope_queries(pope_dir, pope_type, dataset=dataset)

    # Build label lookup
    label_lookup = {}
    for q in queries:
        key = (q['image'], q['question'])
        label_lookup[key] = q['label']

    predictions = []
    labels = []

    for r in results:
        key = (r['image'], r['question'])
        if key in label_lookup:
            # Parse the response if needed
            if 'prediction' in r:
                pred = r['prediction']
            else:
                pred = parse_pope_answer(r.get('response', ''))

            predictions.append(pred)
            labels.append(label_lookup[key])

    return evaluate_pope_results(predictions, labels)


if __name__ == "__main__":
    # Test data loading
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # All POPE data is under datasets/pope/
    pope_dir = os.path.join(project_dir, "datasets", "pope")

    # Image directories for each dataset
    image_dir_map = {
        "coco": os.path.join(project_dir, "datasets", "COCO_2014", "val2014"),
        "aokvqa": os.path.join(project_dir, "datasets", "COCO_2014", "val2014"),  # A-OKVQA uses COCO images
        "gqa": os.path.join(project_dir, "datasets", "gqa", "images"),  # GQA has its own images
    }

    print("Testing POPE data loading...")
    print(f"Supported datasets: {POPE_DATASETS}")
    print(f"POPE dir: {pope_dir}")

    for dataset in POPE_DATASETS:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 40}")

        for pope_type in POPE_TYPES:
            try:
                data = load_pope_data(pope_dir, pope_type, dataset)
                queries = load_pope_queries(pope_dir, pope_type, dataset=dataset)
                print(f"\n  {pope_type}: {len(data)} images, {len(queries)} questions")

                # Check first item
                if data:
                    print(f"    First image: {data[0]['image']}")
                    print(f"    Questions per image: {len(data[0]['questions'])}")
                    print(f"    Sample Q: {data[0]['questions'][0]}")
                    print(f"    Sample A: {data[0]['labels'][0]}")
            except FileNotFoundError as e:
                print(f"\n  {pope_type}: [SKIP] File not found")
            except Exception as e:
                print(f"\n  {pope_type}: [ERROR] {e}")

    # Check if COCO images exist
    print(f"\n{'=' * 40}")
    print("Checking COCO images...")
    coco_image_dir = image_dir_map["coco"]
    if os.path.exists(coco_image_dir):
        try:
            sample_data = load_pope_data(pope_dir, "random", "coco")
            if sample_data:
                sample_img = sample_data[0]['image']
                img_path = os.path.join(coco_image_dir, sample_img)
                print(f"  Sample image path: {img_path}")
                print(f"  Exists: {os.path.exists(img_path)}")
        except:
            print("  Could not load COCO data")
    else:
        print(f"  Image dir not found: {coco_image_dir}")

    # Test parse function
    print(f"\n{'=' * 40}")
    print("Testing parse_pope_answer:")
    test_cases = [
        ("Yes, there is a dog.", "yes"),
        ("No, there is no cat.", "no"),
        ("I can not see any bird.", "no"),
        ("There is definitely a car in the image.", "yes"),
        ("No.", "no"),
        ("Yes.", "yes"),
    ]
    for text, expected in test_cases:
        result = parse_pope_answer(text)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] '{text}' -> '{result}' (expected: '{expected}')")
