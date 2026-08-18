import argparse
import os
import sys
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client_utils import AzureOpenAIClient, OpenAICompatibleClient, SnowflakeCompatibleClient

MIN_CLAIM = 1
MAX_CLAIM = 30

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']

DEFAULT_MODEL = {"azure": "gpt-4-1106-preview", "openai": "gpt-4-1106-preview", "snowflake": "openai-gpt-4.1"}


def flatten_prompt(prompt):
    """Flatten a list of {role, content} chat messages into a single prompt string
    (needed for backends like Snowflake Cortex that don't take a messages array)."""
    return "\n\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in prompt)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--eval_file', default=None, help='filename of the eval_data .json.')
    parser.add_argument('--result_file', required=True, help='filename of the system-generated outputs.')
    
    # claim generation setting
    parser.add_argument('--mode', type=str, default='reference_claims', choices=['reference_claims', 'output_claims', 'input_claims'],
                        help='whether to generate claims for the references, outputs, or inputs')
    parser.add_argument("--use_persection_claims", action="store_true", default=False, help="Generate claims for each section")
    
    # claim generation model
    parser.add_argument('--prompt_file', required=True, help='filename of the prompt dict .json.')
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "snowflake"],
                         help="Backend to send completions to (ignored if --azure is set)")
    parser.add_argument("--model_host", type=str, default=None, help="Host of the OpenAI-compatible endpoint (e.g. vLLM server)")
    parser.add_argument("--model", type=str, default=None, help="Model/deployment name to use (defaults depend on provider)")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max number of new tokens to generate in one step")

    args = parser.parse_args()
    eval_file, result_file, mode, prompt_file, max_new_tokens = args.eval_file, args.result_file, args.mode, args.prompt_file, args.max_new_tokens

    is_snowflake = not args.azure and args.provider == "snowflake"

    if args.azure:
        CLAIM_EXTRACTOR_NAME = CLAIM_EXTRACTOR_DEPLOY_NAME = args.model or DEFAULT_MODEL["azure"]
        # CLAIM_EXTRACTOR_NAME = CLAIM_EXTRACTOR_DEPLOY_NAME = "gpt-35-turbo"
        client = AzureOpenAIClient(model=CLAIM_EXTRACTOR_NAME, deploy_name=CLAIM_EXTRACTOR_DEPLOY_NAME)
    elif is_snowflake:
        CLAIM_EXTRACTOR_NAME = args.model or DEFAULT_MODEL["snowflake"]
        client = SnowflakeCompatibleClient(model=CLAIM_EXTRACTOR_NAME)
    else:
        CLAIM_EXTRACTOR_NAME = args.model or DEFAULT_MODEL["openai"]
        client = OpenAICompatibleClient(model=CLAIM_EXTRACTOR_NAME, model_host=args.model_host)
    
    if mode == 'reference_claims':
        assert eval_file is not None
    
    existing_set = set()
    if mode == 'reference_claims':
        claim_file = eval_file.replace('.json', f'.claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        input_data_file = claim_file if os.path.exists(claim_file) else eval_file
        
        if args.use_persection_claims:
            text_keys = [f"reference_{section}" for section in SECTION_DIVISIONS]
            claim_keys = [f"subclaims_reference_{section}" for section in SECTION_DIVISIONS]
            prompt_template_dict = {f"reference_{section}": json.load(open(prompt_file.replace('persection', section))) for section in SECTION_DIVISIONS}
        else:
            text_keys = ['reference']
            claim_keys = ['subclaims_reference']
            prompt_template_dict = {'reference': json.load(open(prompt_file))}
    
    elif mode=="output_claims":
        claim_file = result_file.replace('.json', f'.output_claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        input_data_file = claim_file if os.path.exists(claim_file) else result_file
        
        if args.use_persection_claims:
            text_keys = [f"output_{section}" for section in SECTION_DIVISIONS]
            claim_keys = [f"subclaims_output_{section}" for section in SECTION_DIVISIONS]
            prompt_template_dict = {f"output_{section}": json.load(open(prompt_file.replace('persection', section))) for section in SECTION_DIVISIONS}
        else:
            text_keys = ['output']
            claim_keys = ['subclaims_output']
            prompt_template_dict = {'output': json.load(open(prompt_file))}

    elif mode == "input_claims":
        claim_file = result_file.replace('.json', f'.input_claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        input_data_file = claim_file if os.path.exists(claim_file) else result_file

        text_keys = ['input']
        claim_keys = ['subclaims_input']
        prompt_template_dict = {'input': json.load(open(prompt_file))}
    data = json.load(open(input_data_file))
            
    for k in prompt_template_dict:
        prompt_template_dict[k][0]['content'] = prompt_template_dict[k][0]['content'].replace('MIN_CLAIM', str(MIN_CLAIM)).replace('MAX_CLAIM', str(MAX_CLAIM))
    if mode == 'reference_claims':
        # copy results from data to result_file 
        result_data = json.load(open(result_file))
        eid2result_item = {x['example_id']:x for x in result_data}
            
    wrong_format_count = 0
    total_count = 0
    for item in data:
        for text_key, claim_key in zip(text_keys, claim_keys):
            if claim_key in item and type(item[claim_key]) == list:
                continue 
            
            if mode == 'reference_claims':
                if item['example_id'] not in eid2result_item:
                    continue
            
            text = item[text_key]
            prompt = deepcopy(prompt_template_dict[text_key])
            prompt[-1]['content'] = text 
            
            if len(text) == 0:
                item[claim_key] = []
                continue
            
            if is_snowflake:
                response = client.completion_with_backoff(prompt=flatten_prompt(prompt), max_tokens=max_new_tokens)
            else:
                response = client.completion_with_backoff(model=client.model, messages=prompt, max_tokens=max_new_tokens)

            try:
                claims_text = response if is_snowflake else response.choices[0].message.content
                claims_text = re.sub(r"^```(json)?|```$", "", claims_text.strip(), flags=re.MULTILINE).strip()
                subclaims_list = re.split('Claim [0-9]+: ', claims_text.replace('\n',''))[1:]
                        
                print(item['example_id'], text_key)
                print('='*50)
                for claim in subclaims_list:
                    print(claim)
                print('='*50)
                item[claim_key] = subclaims_list
            
            except:
                print(f"Wrong format for {item['example_id']}-{text_key}")
                wrong_format_count += 1
                
            total_count += 1
        
            if total_count % 5 == 0:
                print(f'Saving to files: {claim_file}..')
                json.dump(data, open(claim_file, 'w'), indent=4)
                
    if total_count > 0:
        print(f'Saving to files: {claim_file}..')
        json.dump(data, open(claim_file, 'w'), indent=4)


    if mode == 'reference_claims':
        # copy results from data to result_file 
        result_data = json.load(open(result_file))
        eid2item = {x['example_id']:x for x in data}
        for result_item in result_data:
            item = eid2item[result_item['example_id']]
            for claim_key in claim_keys:
                if claim_key in item:
                    result_item[claim_key] = item[claim_key]
                    
        output_file = result_file.replace('.json', f'.claim_min{MIN_CLAIM}max{MAX_CLAIM}.json')
        print(f'Saving to files: {output_file}..')
        json.dump(result_data, open(output_file, 'w'), indent=4)