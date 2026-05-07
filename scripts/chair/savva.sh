# bash scripts/chair/savva.sh

# MODEL: llava, qwen, internvl (space-separated for multiple)
# STRATEGY: savva
# DATASET: chair
# NUM_SAMPLES: number of samples to evaluate (empty = all)

MODELS="llava"           # e.g., "llava qwen internvl"
STRATEGIES="savva"
DATASET=chair
NUM_SAMPLES=

# Run all combinations
for MODEL in $MODELS; do
    mkdir -p logs/chair/$MODEL

    for STRATEGY in $STRATEGIES; do
        CMD="python -u inference.py --model $MODEL --strategy $STRATEGY --dataset $DATASET"
        LOG_NAME="${DATASET}_${MODEL}_${STRATEGY}"

        if [ -n "$NUM_SAMPLES" ]; then
            CMD="$CMD --num-samples $NUM_SAMPLES"
            LOG_NAME="${LOG_NAME}_${NUM_SAMPLES}"
        fi

        echo "Running: $MODEL + $STRATEGY"
        time $CMD > logs/chair/$MODEL/${LOG_NAME}.log 2>&1
        echo "Done: $MODEL + $STRATEGY -> logs/chair/$MODEL/${LOG_NAME}.log"
        echo ""
    done
done
