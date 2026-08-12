MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.33:11435
SHOT=0
TAG="-persection" # TAG="" for full-note generation
DEFAULT_SAVENAME=acibench-test1${TAG}-${MODEL}-shot${SHOT}

SAVENAME=${1:-${DEFAULT_SAVENAME}}



python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name acibench \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model GPT