MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.35:11435
SHOT=6
SAVENAME=${1:-${DEFAULT_SAVENAME}}

python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}