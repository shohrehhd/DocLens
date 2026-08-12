CONFIG_FILE=${1:-configs/aci_gpt41_shot0_snowflake.yaml}
  
 
python run.py --eval_file data/ACI-Bench-TestSet-1_clean.json --result_dir results --dataset_and_tag acibench-test1 \
    --config_file $CONFIG_FILE
