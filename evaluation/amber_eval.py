#!/usr/bin/env python3
"""
AMBER Evaluation Utilities

Complete evaluation pipeline for AMBER benchmark:
- Data loading
- Inference (generative + discriminative)
- Metrics computation (CHAIR, Cover, Hal, Cog, Accuracy, Precision, Recall, F1)
- AMBER score calculation

Author: SAVVA Project
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import set_seed, LogitsProcessorList
from PIL import Image
from tqdm import tqdm

import spacy

# Load spacy model for synonym checking
try:
    nlp = spacy.load("en_core_web_lg")
except:
    nlp = None


def setup_seed(seed=42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def check_synonyms_word(word1, word2, similarity_score=0.8):
    """Check if two words are synonyms using spacy similarity."""
    if nlp is None:
        return False
    token1, token2 = nlp(word1), nlp(word2)
    return token1.similarity(token2) > similarity_score


def extract_nouns(text):
    """Extract nouns from text using NLTK."""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]


def load_amber_eval_data(amber_dir):
    """Load AMBER evaluation data.

    Args:
        amber_dir: Path to AMBER directory

    Returns:
        Dictionary containing:
        - association: Word associations from relation.json
        - hallucination_words: All possible hallucination words
        - global_safe_words: Safe words that shouldn't be marked as hallucination
        - annotations: Ground truth annotations
    """
    relation_path = os.path.join(amber_dir, "data", "relation.json")
    safe_words_path = os.path.join(amber_dir, "data", "safe_words.txt")
    annotations_path = os.path.join(amber_dir, "data", "annotations.json")

    with open(relation_path, 'r') as f:
        association = json.load(f)

    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        hallucination_words.extend(association[word1])

    with open(safe_words_path, 'r') as f:
        global_safe_words = [line.strip() for line in f]

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    return {
        'association': association,
        'hallucination_words': hallucination_words,
        'global_safe_words': global_safe_words,
        'annotations': annotations,
    }


def load_amber_queries(amber_dir, num_samples=None, task_type="generative"):
    """Load AMBER query items.

    Args:
        amber_dir: Path to AMBER directory
        num_samples: Number of samples to load (None = all)
        task_type: "generative", "discriminative", or "all"

    Returns:
        List of query items
    """
    if task_type == "generative":
        query_path = os.path.join(amber_dir, "data", "query", "query_generative.json")
        with open(query_path) as f:
            items = json.load(f)
        items = sorted(
            [it for it in items if isinstance(it.get("id"), int) and 1 <= it["id"] <= 1004],
            key=lambda x: x["id"]
        )
    elif task_type == "discriminative":
        query_path = os.path.join(amber_dir, "data", "query", "query_discriminative.json")
        with open(query_path) as f:
            items = json.load(f)
        items = sorted(items, key=lambda x: x["id"])
    elif task_type == "all":
        query_path = os.path.join(amber_dir, "data", "query", "query_all.json")
        with open(query_path) as f:
            items = json.load(f)
        items = sorted(items, key=lambda x: x["id"])
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if num_samples:
        items = items[:num_samples]
    return items


def evaluate_results(results, eval_data, similarity_score=0.8):
    """Evaluate hallucination metrics using AMBER evaluation.

    Args:
        results: List of dicts with 'id' and 'response' keys
        eval_data: Data from load_amber_eval_data()
        similarity_score: Threshold for synonym matching (default 0.8)

    Returns:
        Dictionary with CHAIR, Cover, Hal, Cog metrics
    """
    association = eval_data['association']
    hallucination_words = eval_data['hallucination_words']
    global_safe_words = eval_data['global_safe_words']
    annotations = eval_data['annotations']

    chair_score, chair_num = 0, 0
    safe_cover_score, safe_cover_num = 0, 0
    hallu_cover_score, hallu_cover_num = 0, 0
    non_hallu_score, non_hallu_num = 0, 0

    for item in results:
        item_id = item['id']
        response = item.get('response', '')
        if not response:
            continue

        gt = annotations[item_id - 1]
        if gt['type'] != 'generative':
            continue

        nouns = extract_nouns(response)
        after_process_nouns = [n for n in nouns if n in hallucination_words]

        safe_words = []
        safe_list = []
        for idx, word in enumerate(gt['truth']):
            safe_words += association.get(word, [])
            safe_list += [idx] * len(association.get(word, []))

        ha_words = []
        ha_list = []
        for idx, word in enumerate(gt['hallu']):
            ha_words += association.get(word, [])
            ha_list += [idx] * len(association.get(word, []))

        safe_words += gt['truth']
        safe_len = len(gt['truth'])
        safe_list += [0] * safe_len
        safe_flag_list = [0] * len(after_process_nouns)

        ha_words += gt['hallu']
        ha_len = len(gt['hallu'])
        ha_list += [0] * ha_len

        for idx, noun in enumerate(after_process_nouns):
            if noun in global_safe_words:
                continue

            if noun in safe_words:
                for j in range(len(safe_words)):
                    if noun == safe_words[j]:
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                continue

            if noun in ha_words:
                for j in range(len(ha_words)):
                    if noun == ha_words[j]:
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break

            for j, check_word in enumerate(ha_words):
                if check_synonyms_word(noun, check_word, similarity_score):
                    if j < (len(ha_list) - ha_len):
                        ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break

            flag = False
            for j, check_word in enumerate(safe_words):
                if check_synonyms_word(noun, check_word, similarity_score):
                    flag = True
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            if flag:
                continue

            safe_flag_list[idx] = 1

        chair_score += sum(safe_flag_list)
        chair_num += len(safe_flag_list)
        safe_cover_score += sum(safe_list[-safe_len:])
        safe_cover_num += len(safe_list[-safe_len:])
        hallu_cover_score += sum(ha_list[-ha_len:])
        hallu_cover_num += len(ha_list[-ha_len:])
        if sum(safe_flag_list) == 0:
            non_hallu_score += 1
        non_hallu_num += 1

    return {
        'CHAIR': round(chair_score / chair_num * 100, 2) if chair_num > 0 else 0.0,
        'Cover': round(safe_cover_score / safe_cover_num * 100, 2) if safe_cover_num > 0 else 0.0,
        'Hal': round(100 - non_hallu_score / non_hallu_num * 100, 2) if non_hallu_num > 0 else 0.0,
        'Cog': round(hallu_cover_score / hallu_cover_num * 100, 2) if hallu_cover_num > 0 else 0.0,
    }


def evaluate_discriminative_results(results, eval_data):
    """Evaluate discriminative task metrics using AMBER evaluation.

    Discriminative tasks are Yes/No questions about:
    - Existence: Is there a [object] in the image?
    - Attribute: state, number, action questions
    - Relation: object relationship questions

    Args:
        results: List of dicts with 'id' and 'response' keys
                 Response should be "Yes" or "No"
        eval_data: Data from load_amber_eval_data()

    Returns:
        Dictionary with discriminative metrics:
        - Accuracy, Precision, Recall, F1 for overall and sub-categories
    """
    annotations = eval_data['annotations']

    # Initialize metrics (using 0.001 to avoid division by zero like original)
    metrics = {
        'qa_correct_score': 0, 'qa_correct_num': 0.001,
        'qa_no_score': 0, 'qa_no_num': 0.001,
        'qa_ans_no_score': 0, 'qa_ans_no_num': 0.001,
        # Existence (discriminative-hallucination)
        'ha_qa_correct_score': 0, 'ha_qa_correct_num': 0.001,
        'ha_qa_no_score': 0, 'ha_qa_no_num': 0.001,
        'ha_qa_ans_no_score': 0, 'ha_qa_ans_no_num': 0.001,
        # Attribute - State
        'as_qa_correct_score': 0, 'as_qa_correct_num': 0.001,
        'as_qa_no_score': 0, 'as_qa_no_num': 0.001,
        'as_qa_ans_no_score': 0, 'as_qa_ans_no_num': 0.001,
        # Attribute - Number
        'an_qa_correct_score': 0, 'an_qa_correct_num': 0.001,
        'an_qa_no_score': 0, 'an_qa_no_num': 0.001,
        'an_qa_ans_no_score': 0, 'an_qa_ans_no_num': 0.001,
        # Attribute - Action
        'aa_qa_correct_score': 0, 'aa_qa_correct_num': 0.001,
        'aa_qa_no_score': 0, 'aa_qa_no_num': 0.001,
        'aa_qa_ans_no_score': 0, 'aa_qa_ans_no_num': 0.001,
        # Relation
        'asso_qa_correct_score': 0, 'asso_qa_correct_num': 0.001,
        'asso_qa_no_score': 0, 'asso_qa_no_num': 0.001,
        'asso_qa_ans_no_score': 0, 'asso_qa_ans_no_num': 0.001,
    }

    # Type to prefix mapping
    type_prefix = {
        'discriminative-hallucination': 'ha',
        'discriminative-attribute-state': 'as',
        'discriminative-attribute-number': 'an',
        'discriminative-attribute-action': 'aa',
        'discriminative-relation': 'asso',
        'relation': 'asso',  # Some annotations use 'relation' type
    }

    for item in results:
        item_id = item['id']
        response = item.get('response', '').strip()

        # Skip empty responses (e.g., skipped oversized images)
        if not response:
            continue

        # Get ground truth
        if item_id - 1 >= len(annotations):
            continue
        gt = annotations[item_id - 1]
        if gt['type'] == 'generative':
            continue  # Skip generative samples

        truth = gt['truth']  # 'yes' or 'no'
        q_type = gt['type']
        prefix = type_prefix.get(q_type, None)

        if prefix is None:
            continue

        # Update count
        metrics['qa_correct_num'] += 1
        metrics[f'{prefix}_qa_correct_num'] += 1

        # Check correctness based on exact match (AMBER original behavior)
        # truth == 'yes' and response == 'Yes', or truth == 'no' and response == 'No'
        if truth == 'yes':
            if response == 'Yes':
                metrics['qa_correct_score'] += 1
                metrics[f'{prefix}_qa_correct_score'] += 1
        else:  # truth == 'no'
            metrics['qa_no_num'] += 1
            metrics[f'{prefix}_qa_no_num'] += 1

            if response == 'No':
                metrics['qa_correct_score'] += 1
                metrics['qa_no_score'] += 1
                metrics[f'{prefix}_qa_correct_score'] += 1
                metrics[f'{prefix}_qa_no_score'] += 1

        # Track "No" answers for precision
        if response == 'No':
            metrics['qa_ans_no_num'] += 1
            metrics[f'{prefix}_qa_ans_no_num'] += 1
            if truth == 'no':
                metrics['qa_ans_no_score'] += 1
                metrics[f'{prefix}_qa_ans_no_score'] += 1

    # Calculate final metrics (matching AMBER original formula)
    # Overall discriminative metrics
    Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 2)
    Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 2)
    Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 2)
    # F1 formula from AMBER: 2 * P * R / (P + R + 0.0001) to avoid division by zero
    F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 2)

    result = {
        'Accuracy': Accuracy,
        'Precision': Precision,
        'Recall': Recall,
        'F1': F1,
    }

    # Sub-category metrics
    for name, prefix in [('Existence', 'ha'), ('Attribute_State', 'as'),
                          ('Attribute_Number', 'an'), ('Attribute_Action', 'aa'),
                          ('Relation', 'asso')]:
        acc = round(metrics[f'{prefix}_qa_correct_score'] / metrics[f'{prefix}_qa_correct_num'] * 100, 2)
        prec = round(metrics[f'{prefix}_qa_ans_no_score'] / metrics[f'{prefix}_qa_ans_no_num'] * 100, 2)
        rec = round(metrics[f'{prefix}_qa_no_score'] / metrics[f'{prefix}_qa_no_num'] * 100, 2)
        f1_sub = round(2 * (prec/100) * (rec/100) / ((prec/100) + (rec/100) + 0.0001) * 100, 2)

        result[f'{name}_Accuracy'] = acc
        result[f'{name}_F1'] = f1_sub

    return result


def compute_amber_score(generative_metrics, discriminative_metrics):
    """Compute AMBER score combining generative and discriminative tasks.

    Formula: AMBER_score = (1 - CHAIR + F1) / 2

    Args:
        generative_metrics: Dict with 'CHAIR' key (percentage, 0-100)
        discriminative_metrics: Dict with 'F1' key (percentage, 0-100)

    Returns:
        AMBER score (percentage, 0-100)
    """
    chair = generative_metrics['CHAIR'] / 100  # Convert to 0-1
    f1 = discriminative_metrics['F1'] / 100    # Convert to 0-1
    amber_score = (1 - chair + f1) / 2 * 100   # Convert back to percentage
    return round(amber_score, 2)


def get_word_associations(word, relation):
    """Get all associated words for a given word from relation dict."""
    associated = set([word])
    if word in relation:
        associated.update(relation[word])
    return associated


def label_tokens_for_hallucination(_tokens, token_texts, gt, relation):
    """Label each token as hallucination or not based on improved CHAIR method.

    Only marks concrete object nouns as hallucination if:
    1. The word is in AMBER's vocabulary (relation.json keys + values)
    2. The word is NOT in the truth list for this image

    Args:
        tokens: Token IDs
        token_texts: Token texts (decoded)
        gt: Ground truth annotation for this image
        relation: Word relation dictionary

    Returns:
        List of (token_idx, is_hallucination, word) tuples
    """
    lemmatizer = WordNetLemmatizer()

    # Build AMBER vocabulary - only these words can be hallucinations
    amber_vocab = set()
    for word in relation.keys():
        amber_vocab.add(word.lower())
        amber_vocab.add(lemmatizer.lemmatize(word.lower()))
        for w in relation[word]:
            amber_vocab.add(w.lower())
            amber_vocab.add(lemmatizer.lemmatize(w.lower()))

    # Build truth word set (with associations)
    truth_words = set()
    for word in gt['truth']:
        truth_words.add(word.lower())
        truth_words.add(lemmatizer.lemmatize(word.lower()))
        if word in relation:
            for w in relation[word]:
                truth_words.add(w.lower())
                truth_words.add(lemmatizer.lemmatize(w.lower()))

    # Reconstruct full response from tokens
    response = ''.join(token_texts)

    # Tokenize the response properly
    try:
        words = nltk.word_tokenize(response)
        tagged = nltk.pos_tag(words)
    except:
        # Fallback
        labels = [(idx, False, None) for idx in range(len(token_texts))]
        return labels

    # Find hallucination words (nouns in AMBER vocab but not in truth)
    hallu_words = set()
    for word, pos in tagged:
        # Only check common nouns (NN, NNS), not proper nouns
        if pos not in ['NN', 'NNS']:
            continue

        lemma = lemmatizer.lemmatize(word.lower())

        # Must be in AMBER vocabulary
        if lemma not in amber_vocab and word.lower() not in amber_vocab:
            continue

        # Must NOT be in truth
        if lemma in truth_words or word.lower() in truth_words:
            continue

        hallu_words.add(lemma)
        hallu_words.add(word.lower())

    # Now map back to tokens
    labels = [(idx, False, None) for idx in range(len(token_texts))]

    # Build token-to-word mapping
    for idx, text in enumerate(token_texts):
        # Clean the token text
        clean = text.lower().strip()
        if not clean:
            continue

        # Remove common prefixes
        clean = clean.lstrip('▁ ')

        # Check if any hallucination word matches
        lemma = lemmatizer.lemmatize(clean)
        if lemma in hallu_words or clean in hallu_words:
            labels[idx] = (idx, True, clean)

    return labels


# =============================================================================
# Inference Functions
# =============================================================================

def normalize_yes_no(response):
    """Normalize response to exact 'Yes' or 'No' for AMBER evaluation."""
    response = response.strip()
    response_lower = response.lower()
    # Check if starts with yes/no
    if response_lower.startswith('yes'):
        return 'Yes'
    elif response_lower.startswith('no'):
        return 'No'
    # Try to find yes/no in the response
    if 'yes' in response_lower:
        return 'Yes'
    elif 'no' in response_lower:
        return 'No'
    # Return original if can't determine (will be marked as wrong by AMBER)
    return response


def run_generative_inference(model, model_name, strategy, strategy_name, items, image_dir,
                              savva_processor_factory=None):
    """Run generative task inference on AMBER dataset.

    Args:
        model: Model instance (LLaVABoostedInterface or QwenBoostedInterface)
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'savva'
        items: List of query items from load_amber_queries()
        image_dir: Path to image directory
        savva_processor_factory: Factory function to create SAVVA logits processor

    Returns:
        List of results with 'id' and 'response' keys
    """
    results = []
    skipped_images = []

    # Max pixels limit for Qwen to prevent OOM (only affects ~16% of AMBER images)
    # Set to None to disable, or e.g. 1280*28*28 (~1M pixels) for safe limit
    QWEN_MAX_PIXELS = 1280 * 28 * 28  # ~1M pixels

    for item in tqdm(items, desc=f"{model_name} Gen {strategy_name}"):
        img_path = os.path.join(image_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            results.append({"id": item["id"], "response": ""})
            continue

        # Skip oversized images to prevent OOM
        if QWEN_MAX_PIXELS is not None:
            w, h = image.size
            if w * h > QWEN_MAX_PIXELS:
                skipped_images.append({"id": item["id"], "image": item["image"], "size": f"{w}x{h}", "pixels": w*h})
                results.append({"id": item["id"], "response": ""})
                tqdm.write(f"  Skipped {item['image']} ({w}x{h}={w*h:,} pixels > {QWEN_MAX_PIXELS:,})")
                continue

        inputs = model.prepare_inputs(image, item["query"])
        if inputs is None:
            results.append({"id": item["id"], "response": ""})
            continue

        visual_indices = model.detect_visual_token_indices(inputs['input_ids'])

        # Set visual indices and reset strategy for SAVVA
        if strategy_name == "savva":
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
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }

        # Add SAVVA logits processor if needed
        if strategy_name == "savva" and savva_processor_factory:
            savva_processor = savva_processor_factory(strategy)
            gen_kwargs["logits_processor"] = LogitsProcessorList([savva_processor])

        with torch.no_grad():
            outputs = model.model.generate(**gen_kwargs)

        if model_name == "llava":
            response = model.processor.tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )
        else:
            response = model.processor.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )

        results.append({"id": item["id"], "response": response})
        tqdm.write(f"  Sample {item['id']}: {response[:100]}...")

    # Report skipped images
    if skipped_images:
        print(f"\n⚠️  Skipped {len(skipped_images)} oversized images (>{QWEN_MAX_PIXELS:,} pixels)")

    return results


def run_discriminative_inference(model, model_name, strategy, strategy_name, items, image_dir,
                                  savva_processor_factory=None):
    """Run discriminative task inference on AMBER dataset.

    Args:
        model: Model instance (LLaVABoostedInterface or QwenBoostedInterface)
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'savva'
        items: List of query items from load_amber_queries()
        image_dir: Path to image directory
        savva_processor_factory: Factory function to create SAVVA logits processor

    Returns:
        List of results with 'id' and 'response' keys
    """
    results = []
    skipped_images = []

    # Max pixels limit for Qwen to prevent OOM (same as generative)
    QWEN_MAX_PIXELS = 1280 * 28 * 28  # ~1M pixels

    for item in tqdm(items, desc=f"{model_name} Disc {strategy_name}"):
        img_path = os.path.join(image_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            results.append({"id": item["id"], "response": ""})
            continue

        # Skip oversized images to prevent OOM
        if QWEN_MAX_PIXELS is not None:
            w, h = image.size
            if w * h > QWEN_MAX_PIXELS:
                skipped_images.append({"id": item["id"], "image": item["image"], "size": f"{w}x{h}", "pixels": w*h})
                results.append({"id": item["id"], "response": ""})
                continue

        # For discriminative task, ask for Yes/No answer
        query = item["query"] + " Please answer Yes or No."
        inputs = model.prepare_inputs(image, query)
        if inputs is None:
            results.append({"id": item["id"], "response": ""})
            continue

        visual_indices = model.detect_visual_token_indices(inputs['input_ids'])

        # Set visual indices and reset strategy for SAVVA
        if strategy_name == "savva":
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
            "max_new_tokens": 10,  # Short answer for Yes/No
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }

        # Add SAVVA logits processor if needed
        if strategy_name == "savva" and savva_processor_factory:
            savva_processor = savva_processor_factory(strategy)
            gen_kwargs["logits_processor"] = LogitsProcessorList([savva_processor])

        with torch.no_grad():
            outputs = model.model.generate(**gen_kwargs)

        if model_name == "llava":
            response = model.processor.tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            ).strip()
        else:
            response = model.processor.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            ).strip()

        # Normalize response to exact 'Yes' or 'No' for AMBER evaluation
        response = normalize_yes_no(response)
        results.append({"id": item["id"], "response": response})

    # Report skipped images
    if skipped_images:
        print(f"\n⚠️  Skipped {len(skipped_images)} oversized images (>{QWEN_MAX_PIXELS:,} pixels)")

    return results


def run_amber_evaluation(model, model_name, strategy, strategy_name, amber_dir,
                          num_gen_samples=None, num_disc_samples=None,
                          savva_processor_factory=None, task_type="all"):
    """Run AMBER evaluation (generative and/or discriminative).

    Args:
        model: Model instance
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'savva'
        amber_dir: Path to AMBER dataset directory
        num_gen_samples: Number of generative samples (None = all)
        num_disc_samples: Number of discriminative samples (None = all)
        savva_processor_factory: Factory function to create SAVVA logits processor
        task_type: "generative", "discriminative", or "all" (default: "all")

    Returns:
        Dictionary with:
        - generative_metrics: CHAIR, Cover, Hal, Cog (if task_type includes generative)
        - discriminative_metrics: Accuracy, Precision, Recall, F1 (if task_type includes discriminative)
        - amber_score: Combined AMBER score (only if both tasks are run)
        - generative_results: Raw results (if task_type includes generative)
        - discriminative_results: Raw results (if task_type includes discriminative)
    """
    image_dir = os.path.join(amber_dir, "image")
    eval_data = load_amber_eval_data(amber_dir)

    run_gen = task_type in ["generative", "all"]
    run_disc = task_type in ["discriminative", "all"]

    gen_metrics = None
    gen_results = None
    disc_metrics = None
    disc_results = None
    amber_score = None

    if run_gen:
        gen_items = load_amber_queries(amber_dir, num_samples=num_gen_samples, task_type="generative")
        print(f"\nLoaded {len(gen_items)} generative samples")

        # Run generative inference
        print(f"\n{'=' * 60}")
        print(f"{model_name.upper()} Generative - {strategy_name.upper()}")
        print("=" * 60)

        gen_results = run_generative_inference(
            model, model_name, strategy, strategy_name,
            gen_items, image_dir, savva_processor_factory
        )
        gen_metrics = evaluate_results(gen_results, eval_data)
        print(f"\nGenerative Results: CHAIR={gen_metrics['CHAIR']:.2f}% Cover={gen_metrics['Cover']:.2f}% "
              f"Hal={gen_metrics['Hal']:.2f}% Cog={gen_metrics['Cog']:.2f}%")

    if run_disc:
        disc_items = load_amber_queries(amber_dir, num_samples=num_disc_samples, task_type="discriminative")
        print(f"\nLoaded {len(disc_items)} discriminative samples")

        # Run discriminative inference
        print(f"\n{'=' * 60}")
        print(f"{model_name.upper()} Discriminative - {strategy_name.upper()}")
        print("=" * 60)

        disc_results = run_discriminative_inference(
            model, model_name, strategy, strategy_name,
            disc_items, image_dir, savva_processor_factory
        )
        disc_metrics = evaluate_discriminative_results(disc_results, eval_data)
        print(f"\nDiscriminative Results: Acc={disc_metrics['Accuracy']:.2f}% Prec={disc_metrics['Precision']:.2f}% "
              f"Rec={disc_metrics['Recall']:.2f}% F1={disc_metrics['F1']:.2f}%")

    # Compute AMBER score only if both tasks are run
    if gen_metrics and disc_metrics:
        amber_score = compute_amber_score(gen_metrics, disc_metrics)

    return {
        'generative_metrics': gen_metrics,
        'discriminative_metrics': disc_metrics,
        'amber_score': amber_score,
        'generative_results': gen_results,
        'discriminative_results': disc_results,
    }
