MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.34:11435
SAVENAME=${1:-${DEFAULT_SAVENAME}}


python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens_role.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file claim_evaluation/prompts/tb_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens_role.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file claim_evaluation/prompts/tb_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens_role.json \
    --result_file results/${SAVENAME}.json \
    --mode input_claims \
    --prompt_file claim_evaluation/prompts/tb_role_label_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

