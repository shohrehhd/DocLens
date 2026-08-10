MODEL=openai-gpt-4.1
SHOT=2
DEFAULT_SAVENAME=meqsum-${MODEL}-shot${SHOT}-quick_test3

SAVENAME=${1:-${DEFAULT_SAVENAME}}

echo "Evaluating ${SAVENAME} .."

# claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --mode claim_recall \
    --prompt_file claim_evaluation/prompts/meqsum_claim_entail.json --provider snowflake

# claim precision
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --mode claim_precision \
    --prompt_file claim_evaluation/prompts/meqsum_claim_entail.json --provider snowflake

# claim same
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --mode same \
    --prompt_file claim_evaluation/prompts/meqsum_claim_entail_same.json --provider snowflake

# eval citations
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name meqsum --split_method sent \
    --prompt_file citation_evaluation/prompts/meqsum_citation_entail.json --provider snowflake

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name meqsum \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model GPT
