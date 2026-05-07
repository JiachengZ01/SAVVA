# SAVVA: Mitigating Hallucinations in LVLMs via Step-wise Adaptive Visual Attention Amplification

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA

```bash
# Install PyTorch first (with CUDA support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data (for CHAIR/AMBER evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet')"

# Download spaCy model (for AMBER evaluation)
python -m spacy download en_core_web_lg
```

## Dataset Preparation

```
datasets/
├── amber/              # AMBER benchmark
│   ├── data/           # Annotations (annotations.json, etc.)
│   └── image/          # Images
├── COCO_2014/          # For POPE-COCO and CHAIR
│   └── val2014/
├── pope/               # POPE annotations
│   ├── coco/           # random/popular/adversarial
│   ├── aokvqa/
│   └── gqa/
├── gqa/                # For POPE-GQA
│   └── images/
├── chair/
│   └── chair.pkl
├── shr/                # SHR benchmark
│   ├── shr_factual_part1.jsonl
│   ├── shr_factual_part2.jsonl
│   └── val_images_final.json
└── VG/                 # For SHR
    └── VG_100K/
```

### AMBER

```bash
# Download from: https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl
cd datasets/amber && unzip AMBER.zip && rm AMBER.zip
```

### COCO val2014

```bash
cd datasets/COCO_2014
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip && rm val2014.zip
```

### POPE

```bash
cd datasets
git clone https://github.com/RUCAIBox/POPE.git pope_repo
mkdir -p pope/coco pope/aokvqa pope/gqa
cp pope_repo/output/coco/*.json pope/coco/
cp pope_repo/output/seem/aokvqa/*.json pope/aokvqa/
cp pope_repo/output/seem/gqa/*.json pope/gqa/
rm -rf pope_repo
```

### GQA

```bash
cd datasets/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip && rm images.zip
```

### SHR

```bash
# Annotations from HA-DPO: https://github.com/zhaozhao99/HA-DPO
# Copy shr_factual_part1.jsonl, shr_factual_part2.jsonl, val_images_final.json to datasets/shr/

# Visual Genome images
cd datasets/VG
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip && rm images.zip
```

## Quick Start

```bash
# AMBER benchmark
bash scripts/amber/savva.sh      # SAVVA
bash scripts/amber/baseline.sh       # Baseline

# POPE benchmark
bash scripts/pope/savva.sh       # SAVVA
bash scripts/pope/baseline.sh        # Baseline

# CHAIR benchmark
bash scripts/chair/savva.sh      # SAVVA
bash scripts/chair/baseline.sh       # Baseline

# SHR benchmark (requires OpenAI API key)
export OPENAI_API_KEY=your_key
bash scripts/shr/savva.sh        # SAVVA
bash scripts/shr/baseline.sh         # Baseline
```

Edit the scripts to change models (e.g., `MODELS="llava qwen internvl"`).

Results are saved to `output/{dataset}/{model}/`, logs to `logs/{dataset}/{model}/`.

## Project Structure

```
SAVVA/
├── configs/
│   └── ours.yaml                 # Model-specific SAVVA parameters
├── scripts/                      # Evaluation scripts
│   ├── amber/
│   ├── pope/
│   ├── chair/
│   └── shr/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py          # Abstract base class
│   ├── no_boost.py               # Baseline (no modification)
│   └── ours/
│       ├── __init__.py
│       ├── savva.py              # SAVVA strategy
│       ├── vge.py                # VGE computation
│       └── logits_processor.py   # Risk update during generation
├── llava_next/
│   └── boosted_interface.py      # LLaVA-NeXT model interface
├── qwen3_vl/
│   └── boosted_interface.py      # Qwen3-VL model interface
├── internvl3_5/
│   └── boosted_interface.py      # InternVL3.5 model interface
├── evaluation/
│   ├── amber_eval.py             # AMBER benchmark
│   ├── pope_eval.py              # POPE benchmark
│   ├── chair_eval.py             # CHAIR benchmark
│   └── shr_eval.py               # SHR benchmark
├── datasets/                     # Benchmark data
├── inference.py                  # Main entry point
└── requirements.txt
```

## Supported Models

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| LLaVA-NeXT-7B | `llava-hf/llava-v1.6-mistral-7b-hf` | 4-bit quantization supported |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B-HF` | |

## License

This project is licensed under the MIT License.
