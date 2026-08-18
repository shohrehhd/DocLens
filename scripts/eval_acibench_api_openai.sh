MODEL=google/gemma-4-31B-it
MODEL_HOST=10.14.29.33:11435
SHOT=0
TAG="-persection" # TAG="" for full-note generation
DEFAULT_SAVENAME=acibench-test1${TAG}-${MODEL}-shot${SHOT}

SAVENAME=${1:-${DEFAULT_SAVENAME}}


# divide reference and output into sections
python data_processing/divide_section.py --result_file data/ACI-Bench-TestSet-1_clean.json
python data_processing/divide_section.py --result_file results/${SAVENAME}.json


python claim_evaluation/generate_subclaims.py --eval_file data/ACI-Bench-TestSet-1_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode reference_claims \
    --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_persection_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


# generate output subclaims
python claim_evaluation/generate_subclaims.py --eval_file data/ACI-Bench-TestSet-1_clean.json \
    --result_file results/${SAVENAME}.json \
    --mode output_claims --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_persection_subclaim_generation.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}


# claim recall
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.claim_min1max30.json \
    --dataset_name acibench --mode claim_recall --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# claim precision
python claim_evaluation/run_entailment.py --result_file results/${SAVENAME}.output_claim_min1max30.json \
    --dataset_name acibench --mode claim_precision --use_persection_claims \
    --prompt_file claim_evaluation/prompts/acibench_claim_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

# eval citations
python citation_evaluation/eval_citation.py --result_file results/${SAVENAME}.json \
    --dataset_name acibench --split_method sent --get_persection_score \
    --prompt_file citation_evaluation/prompts/acibench_persection_citation_entail.json \
    --provider openai \
    --model ${MODEL} \
    --model_host ${MODEL_HOST}

python aggregate_scores.py --result_file results/${SAVENAME}.json --dataset_name acibench \
    --eval_claim_recall --eval_claim_precision --eval_citations --eval_model GPT