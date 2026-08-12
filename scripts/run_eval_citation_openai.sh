MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.33:11435
SAVENAME=${1:-${DEFAULT_SAVENAME}}
# eval citations 
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name mimic --split_method citation \
    --prompt_file citation_evaluation/prompts/mimic_citation_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

