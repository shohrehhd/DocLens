"""Compute input-claim citation coverage: what percentage of a transcript's
utterances were actually cited by at least one of its extracted subclaims_input
(see claim_evaluation/generate_subclaims.py --mode input_claims), i.e. how much
of the raw input the claim extraction step actually drew on.
"""

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', required=True,
                         help='filename of the data with subclaims_input(_metadata) and input_line_metadata.')
    args = parser.parse_args()

    result_file = args.result_file
    savefile = result_file.replace('.json', '.input_citation_coverage.json')

    data = json.load(open(result_file))

    per_example = {}
    skipped = 0
    for item in data:
        eid_str = str(item['example_id'])

        if 'input_line_metadata' not in item or 'subclaims_input_metadata' not in item:
            skipped += 1
            continue

        num_utterances = len(item['input_line_metadata'])
        if num_utterances == 0:
            skipped += 1
            continue

        cited_indices = {idx for meta in item['subclaims_input_metadata'] for idx in meta['citations']}
        num_cited = len(cited_indices)

        per_example[eid_str] = {
            "num_utterances": num_utterances,
            "num_cited": num_cited,
            "coverage": num_cited / num_utterances,
        }

    mean_coverage = sum(x["coverage"] for x in per_example.values()) / len(per_example) if per_example else 0.0

    results = {"per_example": per_example, "mean_coverage": mean_coverage}
    json.dump(results, open(savefile, 'w'), indent=4, sort_keys=True)

    print(f"Saved coverage for {len(per_example)} examples ({skipped} skipped) to {savefile}")
    print(f"Mean utterance coverage: {mean_coverage:.2%}")
