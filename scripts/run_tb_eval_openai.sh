MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.34:11435
SAVENAME=${1:-${DEFAULT_SAVENAME}}


python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --prompt_file claim_evaluation/prompts/tb_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims \
    --prompt_file claim_evaluation/prompts/tb_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

python claim_evaluation/generate_subclaims.py --eval_file data/GU_tb_doclens.json \
    --result_file results/${SAVENAME}.json \
    --mode input_claims \
    --prompt_file claim_evaluation/prompts/tb_input_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


# claim recall
'''
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name tb --mode claim_recall  \
    --prompt_file claim_evaluation/prompts/general_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# claim precision
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name tb --mode claim_precision \
    --prompt_file claim_evaluation/prompts/general_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# claim recall against input claims (output entails input_claims)
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.input_claim_min1max30.json \
    --dataset_name tb --mode claim_recall_input \
    --prompt_file claim_evaluation/prompts/general_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# output claim groundedness (input entails output_claims)
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name tb --mode claim_groundedness \
    --prompt_file claim_evaluation/prompts/general_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# reference claim groundedness (input entails reference_claims)
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name tb --mode reference_groundedness \
    --prompt_file claim_evaluation/prompts/general_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}



python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name tb \
    --eval_claim_recall --eval_claim_precision --eval_claim_recall_input --eval_claim_groundedness --eval_reference_groundedness --eval_model GPT'''