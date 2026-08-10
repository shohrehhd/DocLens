MODEL=openai-gpt-4.1
SAVENAME=${1:-${DEFAULT_SAVENAME}}

python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json \
    --provider snowflake \
    --model ${MODEL} 

python claim_evaluation/generate_subclaims.py --eval_file data/mimic-sampled200_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file claim_evaluation/prompts/mimic_subclaim_generation.json \
    --provider snowflake \
    --model ${MODEL} 