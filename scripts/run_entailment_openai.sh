MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.35:11435
SAVENAME=${1:-${DEFAULT_SAVENAME}}

#claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name mimic --mode claim_recall \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# claim precision 
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name mimic --mode claim_precision \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}
