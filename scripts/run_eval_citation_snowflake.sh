MODEL=openai-gpt-4.1
SAVENAME=${1:-${DEFAULT_SAVENAME}}

# eval citations 
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name mimic \
    --split_method citation \
    --prompt_file citation_evaluation/prompts/mimic_citation_entail.json \
    --provider snowflake \
    --model ${MODEL} 


