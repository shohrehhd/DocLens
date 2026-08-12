SAVENAME=${1:-${DEFAULT_SAVENAME}}
python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name mimic \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model GPT