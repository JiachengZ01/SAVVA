#!/usr/bin/env python3
"""
SHR Evaluation Utilities (Sentence-level Hallucination Ratio)

Complete evaluation pipeline for SHR benchmark.
This implementation strictly follows HA-DPO's original implementation.

Reference: HA-DPO (https://github.com/zhaozhao99/HA-DPO)

Author: AdaVBoost Project
"""

import os
import sys
import json
import time
import copy
import math
import logging
import dataclasses
from typing import Dict, List, Optional, Any, Sequence, Union
from datetime import datetime

from nltk import ngrams
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Constants and Prompts (Exactly from HA-DPO)
# =============================================================================

SHR_PROMPT = "Describe this image in detail."

GPT_JUDGE_PROMPT = '''
Please help me judge if the comment of this image is hallucination or correct.
I will give you a list of region description of a image. The format is [x1, y1, x2, y2]: region description, where [x1, y1, x2, y2] is the bounding box of the region. Highly overlapping bounding boxes may refer to the same object. This is the ground truth information of the image. Besides, I give you some factual information about the content of the image (which is 100% accurate). Your judgement should base on this information. However, this information only descibe the objects in the region of image, so it cannot descibe the subjective part of the image, e.g., atmosphere, style, emotion. In that case, you can return "Cannot judge".
Also, I will give you a list of comments of the image for you to judge if it is hallucination. Please give a judgement one by one along with the reason.

IMPORTANT: You MUST follow this EXACT output format (no markdown, no extra text):
Judgement:
1. hallucination or correct or cannot judge: <reason>
2. hallucination or correct or cannot judge: <reason>
...

Here are the region descriptions of the image:
{}

Factual Information:
{}

Here is the comment for you to judge (hallucination, correct, or cannot judge):
{}
'''


# =============================================================================
# OpenAI API (Exactly from HA-DPO gpt_utils.py, updated for new API)
# =============================================================================

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


# Default decoding args (modified for reproducibility)
decoding_args = OpenAIDecodingArguments(
    temperature=0,  # Set to 0 for deterministic output
    n=1,
    max_tokens=800,
    top_p=1.0,
    stop=["###"],
)


def setup_openai(api_key):
    """Setup OpenAI API (compatible with new openai>=1.0)."""
    import openai
    openai.api_key = api_key
    openai_org = os.getenv("OPENAI_ORG")
    if openai_org is not None:
        openai.organization = openai_org
        logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


def get_gpt_response(prompt, model_name="gpt-5-mini", api_key=None, base_url=None):
    """
    Get GPT response (updated for openai>=1.0 API).

    Exactly replicates HA-DPO's get_gpt_response behavior.
    """
    try:
        from openai import OpenAI
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

        sleep_time = 2
        max_retries = 5
        batch_decoding_args = copy.deepcopy(decoding_args)

        for attempt in range(max_retries):
            try:
                # Build API kwargs based on model type
                is_gpt5 = "gpt-5" in model_name or "o1" in model_name or "o3" in model_name

                if is_gpt5:
                    # GPT-5/o1/o3: Uses Responses API with different parameter format
                    # - input instead of messages
                    # - max_output_tokens instead of max_completion_tokens
                    # - reasoning.effort to control reasoning token usage
                    gpt5_max_tokens = 4096  # High enough for reasoning + output
                    api_kwargs = {
                        "model": model_name,
                        "input": prompt,  # Responses API uses input, not messages
                        "max_output_tokens": gpt5_max_tokens,
                        "reasoning": {
                            "effort": "low",  # Reduce reasoning tokens, leave more for output
                        },
                    }
                    # Debug: log first API call params (without full prompt)
                    if attempt == 0:
                        logging.info(f"GPT-5 API params: model={model_name}, max_output_tokens={gpt5_max_tokens}, reasoning.effort=low")
                else:
                    # GPT-4/GPT-3.5: full parameters supported
                    api_kwargs = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": batch_decoding_args.temperature,
                        "top_p": batch_decoding_args.top_p,
                        "n": batch_decoding_args.n,
                        "stop": batch_decoding_args.stop,
                        "presence_penalty": batch_decoding_args.presence_penalty,
                        "frequency_penalty": batch_decoding_args.frequency_penalty,
                        "max_tokens": batch_decoding_args.max_tokens,
                        "seed": 42,
                    }
                    # logit_bias is tokenizer-specific
                    if "gpt-4" in model_name or "gpt-3" in model_name:
                        api_kwargs["logit_bias"] = {"50256": -100}

                if is_gpt5:
                    # GPT-5: Use Responses API
                    response = client.responses.create(**api_kwargs)
                    # Responses API returns output_text directly
                    content = response.output_text if hasattr(response, 'output_text') else ""
                    if not content:
                        # Try alternative response format
                        if hasattr(response, 'output') and response.output:
                            for item in response.output:
                                if hasattr(item, 'content') and item.content:
                                    for c in item.content:
                                        if hasattr(c, 'text'):
                                            content = c.text
                                            break
                    if not content:
                        logging.warning(f"GPT-5 returned empty response. Full response object: {response}")
                    else:
                        # Debug: print response content
                        print(f"\n[GPT-5 Response]\n{content[:500]}{'...' if len(content) > 500 else ''}\n")
                else:
                    # GPT-4/GPT-3.5: Use Chat Completions API
                    response = client.chat.completions.create(**api_kwargs)
                    content = response.choices[0].message.content
                    if not content:
                        logging.warning(f"GPT returned empty response. Full response object: {response}")
                return content or ""
            except Exception as e:
                logging.warning(f"OpenAIError (attempt {attempt+1}/{max_retries}): {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(
                        batch_decoding_args.max_tokens * 0.8
                    )
                    logging.warning(
                        f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                    )
                elif attempt < max_retries - 1:
                    logging.warning(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                    raise RuntimeError(f"OpenAI API failed after {max_retries} attempts: {e}")
        # If we get here without returning, all retries failed
        raise RuntimeError(f"OpenAI API failed after {max_retries} attempts")
    except ImportError:
        raise ImportError("OpenAI package is required. Install with: pip install openai")


# =============================================================================
# SHR Utils (Exactly from HA-DPO shr_utils.py)
# =============================================================================

def get_model_cap(message):
    """
    Process model response into numbered sentences.
    Exactly from HA-DPO shr_utils.py
    """
    model_cap = message
    model_cap_sep = ""
    cal_all = []
    no = 1
    is_repeated = False
    for i, sentance in enumerate(model_cap.split('.')):
        sentence = sentance.strip()
        # remove repetition
        if sentence in cal_all:
            is_repeated = True
            continue
        if sum([1 for s in cal_all if sentence in s]) > 0 and len(sentence) > 0:
            is_repeated = True
            continue
        if sentence:
            model_cap_sep += f"{no}. {sentence}\n"
            cal_all.append(sentence)
            no += 1
    return model_cap_sep, is_repeated


def get_desc(id2img, id2reg, image_id):
    """
    Get formatted region descriptions for an image.
    Exactly from HA-DPO shr_utils.py
    """
    img_width = id2img[image_id]['width']
    img_height = id2img[image_id]['height']

    description = ""
    for desc in id2reg[image_id]['regions']:
        position = [
            float('%.2f'%f) for f in [
                desc['x']/img_width,
                desc['y']/img_height,
                (desc['x']+ desc['width'])/img_width,
                (desc['y'] + desc['height'])/img_height]
        ]
        phrase = desc['phrase']
        if phrase:
            description += f'{position}: {phrase}\n'

    return description


def seg_cap(message):
    """
    Segment model response into sentences.
    Exactly from HA-DPO shr_utils.py (including the original logic)
    """
    model_cap = message
    cal_all = []
    no = 1
    for i, sentance in enumerate(model_cap.split('.')):
        sentence = sentance.strip()
        # remove repetition
        if sentence in cal_all:
            continue
        if sum([1 for s in cal_all if sentence in s]) > 0:
            continue
        if sentence:
            if sentence[-1] != '.' or sentence[-1] != '?':
                sentence = sentence + '.'
            cal_all.append(sentence)
            no += 1
    return cal_all


def post_process_no_revise(judge, model_response):
    """
    Post-process GPT judgement.
    Based on HA-DPO shr_utils.py with improved robustness for GPT-5.
    """
    model_cap_seg = seg_cap(model_response)

    # segment judgement
    judge_sents = judge.split('\n')
    judge_sents = [sent for sent in judge_sents if len(sent)>0]

    # Find "Judgement:" or "Judgment:" (case-insensitive)
    judge_idx_list = [i for i in range(len(judge_sents))
                      if 'judgement' in judge_sents[i].lower() or 'judgment' in judge_sents[i].lower()]

    if not judge_idx_list:
        raise ValueError(f"Cannot find 'Judgement' in GPT response. Full response:\n{judge}")

    judge_idx = judge_idx_list[0]

    sent_cls = judge_sents[judge_idx+1:]
    sent_cls = [" ".join(sent.split(" ")[1:]).strip() for sent in sent_cls]
    sent_cls = [sent for sent in sent_cls if len(sent)>0]
    cls_res = [sent.split(":")[0].lower() for sent in sent_cls]

    try:
        assert len(model_cap_seg) == len(sent_cls) == len(cls_res)
    except BaseException:
        print(f"error! \njudgement: {judge_sents}\nmodel response: {model_response}")
        # add into annotation
        judge_anno = [
            {
                "model_response": model_cap_seg[i],
                "judgement": None,
                "classification": None,
            } for i in range(len(model_cap_seg))
        ]
        return judge_anno

    # add into annotation
    judge_anno = [
        {
            "model_response": model_cap_seg[i],
            "judgement": sent_cls[i],
            "classification": cls_res[i],
        } for i in range(len(model_cap_seg))
    ]

    return judge_anno


def get_metric(judgement, metrics):
    """
    Compute metrics from judgements.
    Exactly from HA-DPO shr_utils.py
    """
    num_images = len(list(judgement.keys()))
    metrics["num_images"] = num_images
    # avg length (sentence, word)
    total_sent, total_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            total_sent += 1
            total_word += len(judge["model_response"].split(" "))
    metrics["sents_per_image"] = round(total_sent/num_images, 3)
    metrics["words_per_image"] = round(total_word/num_images, 3)
    # avg hallucination (sentence, word)
    total_hal_sent, total_hal_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            if judge["classification"] not in ['hallucination', 'correct', 'cannot judge']:
                continue
            if judge["classification"] == "hallucination":
                total_hal_sent += 1
                total_hal_word += len(judge["model_response"].split(" "))
    metrics["hal_sents_per_image"] = round(total_hal_sent/num_images, 3)
    metrics["hal_words_per_image"] = round(total_hal_word/num_images, 3)
    # ratio of hallucination (sentence, word)
    total_hal_sent, total_hal_word = 0, 0
    total_sent, total_word = 0, 0
    for k in judgement.keys():
        for judge in judgement[k]["judgement"]:
            if judge["classification"] not in ['hallucination', 'correct', 'cannot judge']:
                continue
            if judge["classification"] == "hallucination":
                total_hal_sent += 1
                total_hal_word += len(judge["model_response"].split(" "))
            total_sent += 1
            total_word += len(judge["model_response"].split(" "))
    metrics["hal_sents_ratio"] = round(total_hal_sent/total_sent, 3)
    metrics["hal_words_ratio"] = round(total_hal_word/total_word, 3)
    return metrics


def cal_repetition(sentence, n):
    """
    Calculate n-gram repetition ratio.
    Exactly from HA-DPO shr_utils.py
    """
    sentence = sentence.replace(".", "").replace(":", "").replace("\n", "").replace("?", "").replace(",", "")
    allgrams = ngrams(sentence.split(), n)
    allgrams_list = []
    for gram in allgrams:
        allgrams_list.append(gram)
    return len(list(set(allgrams_list)))/len(allgrams_list)


# =============================================================================
# Data Loading
# =============================================================================

def load_shr_data(shr_path, vg_path):
    """
    Load all SHR evaluation data.
    Exactly replicates HA-DPO json_eval.py data loading.

    Returns:
        val_images: List of validation image info
        id2img: Dict mapping image_id to image metadata
        id2reg: Dict mapping image_id to region descriptions
        id2path: Dict mapping image_id to image path
        factual_inf: Dict mapping image_id to factual information
    """
    # visual genome annotations
    val_images = json.load(open(os.path.join(shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(vg_path, "image_data.json")))
    id2path = {
        _data["image_id"]: os.path.join(vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1])
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]: _data for _data in vg_image_data}
    region = json.load(open(os.path.join(vg_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]: r for r in region}

    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(shr_path, "shr_factual_part2.jsonl")
    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals

    print(f"Loaded {len(val_images)} validation images")
    print(f"Loaded {len(id2img)} VG image metadata")
    print(f"Loaded {len(id2reg)} region descriptions")
    print(f"Loaded {len(factual_inf)} factual annotations")

    return val_images, id2img, id2reg, id2path, factual_inf


def load_shr_queries(shr_dir, vg_dir, num_samples=None):
    """
    Load SHR queries for inference.

    Returns:
        List of dicts with image_id and image_path
    """
    val_images = json.load(open(os.path.join(shr_dir, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(vg_dir, "image_data.json")))

    id2path = {
        _data["image_id"]: os.path.join(vg_dir, _data["url"].split("/")[-2], _data["url"].split("/")[-1])
        for _data in vg_image_data
    }

    queries = []
    for item in val_images:
        image_id = item["image_id"]
        if image_id in id2path:
            image_path = id2path[image_id]
            if os.path.exists(image_path):
                queries.append({
                    "image_id": image_id,
                    "image_path": image_path,
                })

    if num_samples is not None:
        queries = queries[:num_samples]

    print(f"Loaded {len(queries)} SHR queries")
    return queries


# =============================================================================
# Main Evaluation Function (Exactly replicates HA-DPO json_eval.py)
# =============================================================================

def evaluate_shr(
    json_file: Union[str, Dict],
    shr_path: str,
    vg_path: str,
    api_key: str,
    base_url: Optional[str] = None,
    model_name: str = "gpt-5-mini",
    no_gpt_judge: bool = False,
    output_dir: Optional[str] = None,
):
    """
    Evaluate SHR metrics.
    Exactly replicates HA-DPO json_eval.py main function.

    Args:
        json_file: Path to JSON file or dict with format {image_id: caption}
        shr_path: Path to SHR annotation directory
        vg_path: Path to Visual Genome directory
        api_key: OpenAI API key
        base_url: Optional OpenAI API base URL
        model_name: LLM judge model name (default: gpt-5-mini)
        no_gpt_judge: If True, only compute repetition metrics
        output_dir: Optional output directory for results

    Returns:
        Dict with metrics and judgements
    """
    # Load data
    val_images, id2img, id2reg, id2path, factual_inf = load_shr_data(shr_path, vg_path)

    # Load json file
    if isinstance(json_file, str):
        json_data = json.load(open(json_file))
    else:
        json_data = json_file

    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}

    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0

    # Only evaluate images that have captions in json_data
    eval_images = [img for img in val_images if str(img["image_id"]) in json_data]
    print(f"Evaluating {len(eval_images)} images (out of {len(val_images)} total)")

    for _data in tqdm(eval_images):
        image_id = _data["image_id"]

        # ask model to describe the image
        model_response = json_data[str(image_id)]

        # Get GPT judgement inputs
        description = get_desc(id2img, id2reg, int(image_id))
        model_cap_sep, is_repeated = get_model_cap(model_response)

        # Calculate repetition
        gram1 = cal_repetition(model_response, 1)
        gram2 = cal_repetition(model_response, 2)
        gram3 = cal_repetition(model_response, 3)
        gram4 = cal_repetition(model_response, 4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4

        # Skip GPT judgement if requested
        if no_gpt_judge:
            continue

        # GPT judgement
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"

        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)

        for run in run_all:
            retry_count = 0
            while True:
                judge = get_gpt_response(prompt=judge_prompt, model_name=model_name, api_key=api_key, base_url=base_url)
                # Accept both British "Judgement" and American "Judgment" spelling
                if "Judgement" in judge or "Judgment" in judge:
                    break
                retry_count += 1
                if retry_count >= 3:
                    logging.warning(f"GPT response missing 'Judgement' after {retry_count} retries. Response: {judge[:200]}...")
                    break  # Give up after 3 retries to avoid infinite loop
            # post-process
            final_judge = post_process_no_revise(judge, model_response)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "mode_response": model_response,
                "judgement": final_judge,
            }

    whole_sample_cnt = len(eval_images)

    if no_gpt_judge:
        metrics = {
            'gram-1-repetition': round(_gram1/whole_sample_cnt, 3),
            'gram-2-repetition': round(_gram2/whole_sample_cnt, 3),
            'gram-3-repetition': round(_gram3/whole_sample_cnt, 3),
            'gram-4-repetition': round(_gram4/whole_sample_cnt, 3),
        }
        print(f"gram-1 repetition: {metrics['gram-1-repetition']}")
        print(f"gram-2 repetition: {metrics['gram-2-repetition']}")
        print(f"gram-3 repetition: {metrics['gram-3-repetition']}")
        print(f"gram-4 repetition: {metrics['gram-4-repetition']}")
        return {'metrics': metrics, 'judgement': None}

    # Compute metrics
    metrics = {}
    for run in run_all:
        metrics[run] = {}
        get_metric(judgement[run], metrics[run])

    # repetition
    metrics['gram-1-repetition'] = round(_gram1/whole_sample_cnt, 3)
    metrics['gram-2-repetition'] = round(_gram2/whole_sample_cnt, 3)
    metrics['gram-3-repetition'] = round(_gram3/whole_sample_cnt, 3)
    metrics['gram-4-repetition'] = round(_gram4/whole_sample_cnt, 3)

    # hallucination ratio (main metric)
    metrics["mean_hal_ratio"] = round(
        sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
    )

    # Print results
    print(f"\n{'='*60}")
    print("SHR Evaluation Results")
    print(f"{'='*60}")
    print(f"mean_hal_ratio (SHR): {metrics['mean_hal_ratio']}")
    print(f"hal_sents_ratio: {metrics['run1']['hal_sents_ratio']}")
    print(f"hal_words_ratio: {metrics['run1']['hal_words_ratio']}")
    print(f"sents_per_image: {metrics['run1']['sents_per_image']}")
    print(f"words_per_image: {metrics['run1']['words_per_image']}")
    print(f"gram-1 repetition: {metrics['gram-1-repetition']}")
    print(f"gram-2 repetition: {metrics['gram-2-repetition']}")
    print(f"gram-3 repetition: {metrics['gram-3-repetition']}")
    print(f"gram-4 repetition: {metrics['gram-4-repetition']}")
    print(f"{'='*60}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        localtime = time.asctime(time.localtime(time.time())).replace(' ', '_')
        eval_path = os.path.join(output_dir, localtime)
        os.makedirs(eval_path, exist_ok=True)

        # dump judgement file
        with open(os.path.join(eval_path, 'judgement.json'), "w") as f:
            json.dump(judgement, f)
        # dump metric file
        with open(os.path.join(eval_path, 'metrics.json'), "w") as f:
            json.dump(metrics, f)

        print(f"Results saved to: {eval_path}")

    return {'metrics': metrics, 'judgement': judgement}


# =============================================================================
# Inference Helper (for AdaVBoost models)
# =============================================================================

def run_shr_inference(
    model,
    shr_dir: str,
    vg_dir: str,
    num_samples: Optional[int] = None,
    strategy=None,
    output_file: Optional[str] = None,
):
    """
    Run model inference for SHR evaluation.

    Args:
        model: Model interface with prepare_inputs method
        shr_dir: Path to SHR data
        vg_dir: Path to VG data
        num_samples: Number of samples (None = all 200)
        strategy: Optional AdaVBoost strategy
        output_file: Path to save captions JSON

    Returns:
        Dict with format {image_id: caption}
    """
    from PIL import Image
    import torch
    from transformers import LogitsProcessorList

    # Load queries
    queries = load_shr_queries(shr_dir, vg_dir, num_samples=num_samples)

    # Setup AdaVBoost processor if strategy provided
    adavboost_processor = None
    if strategy is not None:
        from strategies.ours import RiskLogitsProcessor
        adavboost_processor = RiskLogitsProcessor(strategy)

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
        if strategy is not None:
            visual_indices = model.detect_visual_token_indices(inputs['input_ids'])
            model._visual_indices_tensor = torch.tensor(
                visual_indices, dtype=torch.long, device=model.device
            )
            strategy.reset()
            if adavboost_processor:
                adavboost_processor.step_count = 0
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

        # Generate (using SHR default settings: num_beams=5)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 5,  # SHR default
            "pad_token_id": pad_token_id,
        }

        if adavboost_processor:
            gen_kwargs["logits_processor"] = LogitsProcessorList([adavboost_processor])

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

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Captions saved to: {output_file}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHR Evaluation")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, help="OpenAI API base URL")
    parser.add_argument("--model-name", type=str, default="gpt-5-mini", help="LLM judge model name (default: gpt-5-mini)")
    parser.add_argument("--json-file", type=str, help="Path to JSON file with model responses")
    parser.add_argument("--vg-path", type=str, default="datasets/VG", help="Path to VG data")
    parser.add_argument("--shr-path", type=str, default="datasets/shr", help="Path to SHR data")
    parser.add_argument("--no-gpt-judge", action='store_true', help="Skip GPT evaluation")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--test", action='store_true', help="Run test mode")
    args = parser.parse_args()

    if args.test:
        # Test data loading
        print("Testing data loading...")
        val_images, id2img, id2reg, id2path, factual_inf = load_shr_data(args.shr_path, args.vg_path)

        # Test get_desc
        sample_id = val_images[0]["image_id"]
        desc = get_desc(id2img, id2reg, sample_id)
        print(f"\nSample region description (image {sample_id}):")
        print(desc[:500] + "..." if len(desc) > 500 else desc)

        # Test factual info
        if str(sample_id) in factual_inf:
            print(f"\nSample factual info:")
            for fact in factual_inf[str(sample_id)][:3]:
                print(f"  - {fact}")
    else:
        if not args.json_file:
            parser.error("--json-file is required for evaluation")
        if not args.api_key and not args.no_gpt_judge:
            parser.error("--api-key is required for GPT evaluation")

        evaluate_shr(
            json_file=args.json_file,
            shr_path=args.shr_path,
            vg_path=args.vg_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            no_gpt_judge=args.no_gpt_judge,
            output_dir=args.output_dir,
        )
