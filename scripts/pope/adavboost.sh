# bash scripts/pope/adavboost.sh

# MODEL: llava, qwen, internvl (space-separated for multiple)
# STRATEGY: adavboost
# DATASET: pope
# POPE_DATASET: coco, aokvqa, gqa, all
# POPE_TYPE: random, popular, adversarial, all
# NUM_SAMPLES: number of samples to evaluate (empty = all)

MODELS="llava"           # e.g., "llava qwen internvl"
STRATEGIES="adavboost"
DATASET=pope
POPE_DATASET=all
POPE_TYPE=all
NUM_SAMPLES=

# Run all combinations
for MODEL in $MODELS; do
    mkdir -p logs/pope/$MODEL

    for STRATEGY in $STRATEGIES; do
        CMD="python -u inference.py --model $MODEL --strategy $STRATEGY --dataset $DATASET --pope-dataset $POPE_DATASET --pope-type $POPE_TYPE"
        LOG_NAME="${DATASET}_${POPE_DATASET}_${MODEL}_${STRATEGY}_${POPE_TYPE}"

        if [ -n "$NUM_SAMPLES" ]; then
            CMD="$CMD --num-samples $NUM_SAMPLES"
            LOG_NAME="${LOG_NAME}_${NUM_SAMPLES}"
        fi

        echo "Running: $MODEL + $STRATEGY (${POPE_DATASET})"
        time $CMD > logs/pope/$MODEL/${LOG_NAME}.log 2>&1
        echo "Done: $MODEL + $STRATEGY -> logs/pope/$MODEL/${LOG_NAME}.log"
        echo ""
    done
done
