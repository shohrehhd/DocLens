MODEL=openai-gpt-4.1
SAVENAME=${1:-${DEFAULT_SAVENAME}}

#claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name mimic --mode claim_recall \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json \
    --provider snowflake \
    --model ${MODEL} 

# claim precision 
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name mimic --mode claim_precision \
    --prompt_file claim_evaluation/prompts/mimic_claim_entail.json \
    --provider snowflake \
    --model ${MODEL} 