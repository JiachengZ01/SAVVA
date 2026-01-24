#!/usr/bin/env python3
"""
CHAIR Evaluation Utilities

Complete evaluation pipeline for CHAIR (Caption Hallucination Assessment with Image Relevance) benchmark:
- Data loading (COCO val2014 images)
- Inference (image captioning)
- Metrics computation (CHAIRs, CHAIRi, Recall, Precision, F1)

The evaluation logic follows the original CHAIR implementation:
Reference: https://github.com/LisaAnne/Hallucination

Author: AdaVBoost Project
"""

import os
import json
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import set_seed, LogitsProcessorList


# =============================================================================
# MSCOCO Synonyms (from CHAIR original)
# =============================================================================

SYNONYMS_TXT = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


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


# =============================================================================
# CHAIR Evaluator Class
# =============================================================================

class CHAIREvaluator:
    """CHAIR evaluation class for measuring object hallucination in image captions."""

    def __init__(self, cache_path=None):
        """Initialize CHAIR evaluator.

        Args:
            cache_path: Path to pre-cached evaluator pickle file.
                       If provided and exists, loads from cache.
                       If not provided, builds from synonyms only (no COCO annotations).
        """
        self.imid_to_objects = defaultdict(set)

        # Parse synonyms
        synonyms = SYNONYMS_TXT.strip().splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms if s.strip()]

        self.mscoco_objects = []  # All MSCOCO objects and synonyms
        self.inverse_synonym_dict = {}

        for synonym_list in synonyms:
            self.mscoco_objects.extend(synonym_list)
            for s in synonym_list:
                self.inverse_synonym_dict[s] = synonym_list[0]

        # Double words that should be treated as single words
        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
            'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
            'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
            'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear',
            'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven',
            'home plate', 'train track'
        ]

        # Animal and vehicle qualifiers
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                        'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict[f'baby {animal_word}'] = animal_word
            self.double_word_dict[f'adult {animal_word}'] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f'passenger {vehicle_word}'] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        # Load from cache if available
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)

    def _load_from_cache(self, cache_path):
        """Load pre-computed imid_to_objects from cache.

        We use a custom unpickler to redirect the class lookup for compatibility.
        """
        class CHAIRUnpickler(pickle.Unpickler):
            """Custom unpickler that redirects CHAIR class."""
            def find_class(unpickler_self, module, name):
                # Redirect __main__.CHAIR or any CHAIR class to CHAIREvaluator
                if name == 'CHAIR':
                    return CHAIREvaluator
                return super().find_class(module, name)

        with open(cache_path, 'rb') as f:
            cached = CHAIRUnpickler(f).load()
            if hasattr(cached, 'imid_to_objects'):
                self.imid_to_objects = cached.imid_to_objects
                print(f"Loaded CHAIR cache with {len(self.imid_to_objects)} images")

    def get_wordnet_pos(self, tag):
        """Convert POS tag to WordNet POS."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
        """Extract MSCOCO object words from caption.

        Args:
            caption: Caption string

        Returns:
            Tuple of (words, node_words, idxs, raw_words)
            - words: MSCOCO words found in caption
            - node_words: Canonical form of each word
            - idxs: Indices of words in caption
            - raw_words: All words after preprocessing
        """
        # Standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        words = lemmas_sent

        # Replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i+2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # Handle "toilet seat" special case
        if ('toilet' in words) and ('seat' in words):
            words = [word for word in words if word != 'seat']

        # Get only MSCOCO objects
        mscoco_set = set(self.mscoco_objects)
        idxs = [idx for idx, word in enumerate(words) if word in mscoco_set]
        words = [word for word in words if word in mscoco_set]

        # Map to canonical form
        node_words = [self.inverse_synonym_dict[word] for word in words]

        return words, node_words, idxs, double_words

    def compute_chair(self, results):
        """Compute CHAIR metrics for generated captions.

        Args:
            results: List of dicts with 'image_id' and 'caption' keys

        Returns:
            Dictionary with overall metrics and per-sentence details
        """
        num_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0
        len_caps = 0.0

        num_recall_gt_objects = 0.0
        num_gt_objects = 0.0
        num_generated_objects = 0.0

        output = {'sentences': []}

        for item in tqdm(results, desc="Computing CHAIR"):
            cap = item.get('caption', '')
            imid = item.get('image_id')

            if not cap or imid is None:
                continue

            # Get words from caption
            words, node_words, idxs, raw_words = self.caption_to_words(cap)

            # Get ground truth objects for this image
            gt_objects = self.imid_to_objects.get(imid, set())

            cap_dict = {
                'image_id': imid,
                'caption': cap,
                'mscoco_hallucinated_words': [],
                'mscoco_gt_words': list(gt_objects),
                'mscoco_generated_words': list(node_words),
                'hallucination_idxs': [],
                'words': raw_words,
                'metrics': {
                    'CHAIRs': 0,
                    'CHAIRi': 0.0,
                    'Recall': 0.0,
                    'Precision': 0.0,
                    'F1': 0.0,
                }
            }

            # Count hallucinated words
            coco_word_count += len(node_words)
            hallucinated = False
            recall_gt_objects = set()

            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)

            # Count hallucinated caps
            num_caps += 1
            len_caps += len(raw_words)
            if hallucinated:
                num_hallucinated_caps += 1

            # Per-image metrics
            num_gt_objects += len(gt_objects)
            num_generated_objects += len(set(node_words))
            num_recall_gt_objects += len(recall_gt_objects)

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)

            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words))

            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
                if len(node_words) == 0:
                    cap_dict['metrics']['Precision'] = 0.0
                else:
                    cap_dict['metrics']['Precision'] = len(recall_gt_objects) / len(set(node_words))

                p = cap_dict['metrics']['Precision']
                r = cap_dict['metrics']['Recall']
                if (p + r) > 0:
                    cap_dict['metrics']['F1'] = 2 * p * r / (p + r)

            output['sentences'].append(cap_dict)

        # Overall metrics
        if num_caps > 0 and coco_word_count > 0:
            chair_s = num_hallucinated_caps / num_caps
            chair_i = hallucinated_word_count / coco_word_count
            recall = num_recall_gt_objects / num_gt_objects if num_gt_objects > 0 else 0
            precision = num_recall_gt_objects / num_generated_objects if num_generated_objects > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
            avg_len = len_caps / num_caps

            output['overall_metrics'] = {
                'CHAIRs': round(chair_s * 100, 2),
                'CHAIRi': round(chair_i * 100, 2),
                'Recall': round(recall * 100, 2),
                'Precision': round(precision * 100, 2),
                'F1': round(f1 * 100, 2),
                'Len': round(avg_len, 1),
                'num_caps': int(num_caps),
            }
        else:
            output['overall_metrics'] = {
                'CHAIRs': 0.0,
                'CHAIRi': 0.0,
                'Recall': 0.0,
                'Precision': 0.0,
                'F1': 0.0,
                'Len': 0.0,
                'num_caps': 0,
            }

        return output


# =============================================================================
# Data Loading
# =============================================================================

def load_coco_image_ids(coco_dir):
    """Load all image IDs from COCO val2014 directory.

    Args:
        coco_dir: Path to COCO directory (containing val2014/)

    Returns:
        List of image IDs (integers)
    """
    image_dir = os.path.join(coco_dir, "val2014")
    image_ids = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Extract image ID from filename: COCO_val2014_000000123456.jpg -> 123456
            try:
                imid = int(filename.split('_')[-1].split('.')[0])
                image_ids.append(imid)
            except ValueError:
                continue

    return sorted(image_ids)


CHAIR_DEFAULT_SAMPLES = 500  # Default number of samples for CHAIR evaluation


def load_chair_queries(coco_dir, num_samples=None, seed=42):
    """Load CHAIR evaluation queries (COCO images).

    Args:
        coco_dir: Path to COCO directory
        num_samples: Number of samples to load (None = use CHAIR_DEFAULT_SAMPLES)
        seed: Random seed for reproducible sampling

    Returns:
        List of dicts with 'image_id' and 'image_path' keys
    """
    image_dir = os.path.join(coco_dir, "val2014")
    image_ids = load_coco_image_ids(coco_dir)

    # Default to CHAIR_DEFAULT_SAMPLES for efficiency
    if num_samples is None:
        num_samples = CHAIR_DEFAULT_SAMPLES

    # Random sampling with fixed seed for reproducibility
    if num_samples < len(image_ids):
        rng = random.Random(seed)
        image_ids = rng.sample(image_ids, num_samples)
        image_ids = sorted(image_ids)  # Sort for consistent ordering

    queries = []
    for imid in image_ids:
        filename = f"COCO_val2014_{imid:012d}.jpg"
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            queries.append({
                'image_id': imid,
                'image_path': image_path,
            })

    return queries


# =============================================================================
# Inference Functions
# =============================================================================

# Default prompt for image captioning
CHAIR_PROMPT = "Please help me describe the image in detail."


def run_chair_inference(model, model_name, strategy, strategy_name, queries,
                        adavboost_processor_factory=None, prompt=None):
    """Run CHAIR inference (image captioning).

    Args:
        model: Model instance (LLaVABoostedInterface or QwenBoostedInterface)
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'adavboost'
        queries: List of query items from load_chair_queries()
        adavboost_processor_factory: Factory function to create AdaVBoost logits processor
        prompt: Prompt for captioning (default: CHAIR_PROMPT)

    Returns:
        List of results with 'image_id' and 'caption' keys
    """
    if prompt is None:
        prompt = CHAIR_PROMPT

    results = []
    skipped_images = []

    # Max pixels limit for Qwen to prevent OOM
    QWEN_MAX_PIXELS = 1280 * 28 * 28  # ~1M pixels

    for item in tqdm(queries, desc=f"{model_name} CHAIR {strategy_name}"):
        image_path = item["image_path"]
        image_id = item["image_id"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            results.append({
                "image_id": image_id,
                "caption": "",
                "skipped": True,
            })
            continue

        # Skip oversized images to prevent OOM
        if QWEN_MAX_PIXELS is not None:
            w, h = image.size
            if w * h > QWEN_MAX_PIXELS:
                skipped_images.append({
                    "image_id": image_id,
                    "size": f"{w}x{h}",
                    "pixels": w * h
                })
                results.append({
                    "image_id": image_id,
                    "caption": "",
                    "skipped": True,
                })
                continue

        # Prepare inputs
        inputs = model.prepare_inputs(image, prompt)
        if inputs is None:
            results.append({
                "image_id": image_id,
                "caption": "",
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
            "max_new_tokens": 512,
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
            caption = model.processor.tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )
        else:
            caption = model.processor.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )

        results.append({
            "image_id": image_id,
            "caption": caption,
        })

    # Report skipped images
    if skipped_images:
        print(f"\n  Skipped {len(skipped_images)} oversized images (>{QWEN_MAX_PIXELS:,} pixels)")

    return results


def run_chair_evaluation(model, model_name, strategy, strategy_name,
                         coco_dir, cache_path=None, num_samples=None,
                         adavboost_processor_factory=None):
    """Run complete CHAIR evaluation.

    Args:
        model: Model instance
        model_name: 'llava' or 'qwen'
        strategy: Strategy instance
        strategy_name: 'baseline' or 'adavboost'
        coco_dir: Path to COCO directory (containing val2014/)
        cache_path: Path to CHAIR evaluator cache (chair.pkl)
        num_samples: Number of samples (None = use CHAIR_DEFAULT_SAMPLES=500)
        adavboost_processor_factory: Factory function to create AdaVBoost logits processor

    Returns:
        Dictionary with metrics and raw results
    """
    # Get total image count first
    total_images = len(load_coco_image_ids(coco_dir))
    queries = load_chair_queries(coco_dir, num_samples)

    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} CHAIR - {strategy_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Sampled {len(queries)} images (from {total_images} total, seed=42)")

    # Run inference
    results = run_chair_inference(
        model, model_name, strategy, strategy_name,
        queries, adavboost_processor_factory
    )

    # Filter out skipped results for evaluation
    valid_results = [r for r in results if not r.get('skipped', False)]
    skipped_count = len(results) - len(valid_results)

    if skipped_count > 0:
        print(f"\n  Skipped {skipped_count} samples (not included in metrics)")

    # Create evaluator and compute metrics
    evaluator = CHAIREvaluator(cache_path=cache_path)
    chair_output = evaluator.compute_chair(valid_results)

    metrics = chair_output['overall_metrics']

    print(f"\nResults (n={metrics['num_caps']}):")
    print(f"  CHAIRs:    {metrics['CHAIRs']:.2f}% (lower is better)")
    print(f"  CHAIRi:    {metrics['CHAIRi']:.2f}% (lower is better)")
    print(f"  Recall:    {metrics['Recall']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.2f}%")
    print(f"  F1:        {metrics['F1']:.2f}%")
    print(f"  Avg Len:   {metrics['Len']:.1f} words")

    return {
        'metrics': metrics,
        'results': results,
        'sentences': chair_output['sentences'],
    }


# =============================================================================
# Standalone Evaluation
# =============================================================================

def evaluate_saved_results(results_file, cache_path):
    """Evaluate saved inference results.

    Args:
        results_file: Path to JSON/JSONL file with results
        cache_path: Path to CHAIR evaluator cache

    Returns:
        Metrics dictionary
    """
    # Load results
    ext = os.path.splitext(results_file)[-1]
    if ext == '.json':
        with open(results_file, 'r') as f:
            results = json.load(f)
    elif ext == '.jsonl':
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Evaluate
    evaluator = CHAIREvaluator(cache_path=cache_path)
    chair_output = evaluator.compute_chair(results)

    return chair_output['overall_metrics']


if __name__ == "__main__":
    # Test data loading
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    coco_dir = os.path.join(project_dir, "datasets", "COCO_2014")
    cache_path = os.path.join(project_dir, "datasets", "chair", "chair.pkl")

    print("Testing CHAIR data loading...")
    print(f"COCO dir: {coco_dir}")
    print(f"Cache path: {cache_path}")

    # Test loading image IDs
    image_ids = load_coco_image_ids(coco_dir)
    print(f"\nFound {len(image_ids)} images in val2014")

    if image_ids:
        print(f"Sample IDs: {image_ids[:5]}")

    # Test loading queries
    queries = load_chair_queries(coco_dir, num_samples=10)
    print(f"\nLoaded {len(queries)} queries")
    if queries:
        print(f"Sample query: {queries[0]}")

    # Test evaluator
    print("\nTesting CHAIR evaluator...")
    evaluator = CHAIREvaluator(cache_path=cache_path)
    print(f"Evaluator has {len(evaluator.imid_to_objects)} images with annotations")

    # Test caption processing
    test_caption = "A dog is sitting on a couch next to a cat."
    words, node_words, idxs, raw_words = evaluator.caption_to_words(test_caption)
    print(f"\nTest caption: {test_caption}")
    print(f"MSCOCO words: {words}")
    print(f"Node words: {node_words}")
