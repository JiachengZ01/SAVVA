# bash scripts/shr/adavboost.sh

# MODEL: llava, qwen, internvl (space-separated for multiple)
# STRATEGY: adavboost
# DATASET: shr
# NUM_SAMPLES: number of samples to evaluate (empty = all)
# Note: SHR requires OpenAI API key for LLM-as-a-judge evaluation

MODELS="llava"           # e.g., "llava qwen internvl"
STRATEGIES="adavboost"
DATASET=shr
NUM_SAMPLES=

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Usage: export OPENAI_API_KEY=your_key && bash scripts/shr/adavboost.sh"
    exit 1
fi

# Run all combinations
for MODEL in $MODELS; do
    mkdir -p logs/shr/$MODEL

    for STRATEGY in $STRATEGIES; do
        CMD="python -u inference.py --model $MODEL --strategy $STRATEGY --dataset $DATASET --shr-api-key $OPENAI_API_KEY"
        LOG_NAME="${DATASET}_${MODEL}_${STRATEGY}"

        if [ -n "$NUM_SAMPLES" ]; then
            CMD="$CMD --num-samples $NUM_SAMPLES"
            LOG_NAME="${LOG_NAME}_${NUM_SAMPLES}"
        fi

        echo "Running: $MODEL + $STRATEGY"
        time $CMD > logs/shr/$MODEL/${LOG_NAME}.log 2>&1
        echo "Done: $MODEL + $STRATEGY -> logs/shr/$MODEL/${LOG_NAME}.log"
        echo ""
    done
done
