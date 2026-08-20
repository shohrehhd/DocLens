"""Convert per-patient transcript turns + human summaries into DocLens format.

Input:
  - a directory of transcript files, each named "<transcript_id>.json" and
    containing a list of turns: {"start", "end", "speaker", "text", "logprob"}.
  - a patients file: {"patients": [{"label", "transcript", "human_summary",
    "segments": [{"start", "end"}], ...}]}, where "transcript" names the
    transcript file (without extension) and "segments" gives the time
    window(s) of that transcript belonging to this patient.

Output: a list of DocLens-style records:
  {"input", "reference", "example_id", "section", "orig_line_id",
   "input_line_metadata"}
where "input" is newline-joined utterances formatted as "[i][Speaker] Text"
(matching the ACI-Bench convention), e.g.:
  [0][Doctor] I am presenting Mr x who is a 73 year old gentleman.
and "input_line_metadata" is a list, aligned with those "[i]" markers, of
each utterance's speaker -- so that claims later extracted from "input"
(see claim_evaluation/generate_subclaims.py) can inherit it via their
citation back to line i.
"""

import argparse
import json
import os


def parse_timestamp(ts):
    """Convert "HH:MM:SS.mmm" to seconds."""
    hours, minutes, seconds = ts.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def turn_in_segments(turn, segments):
    if not segments:
        return True
    turn_start = parse_timestamp(turn["start"])
    return any(
        parse_timestamp(seg["start"]) <= turn_start <= parse_timestamp(seg["end"])
        for seg in segments
    )


def build_input(turns):
    lines = []
    for i, turn in enumerate(turns):
        lines.append("[%d][%s] %s" % (i, turn.get("speaker", ""), turn.get("text", "").strip()))
    return "\n".join(lines)


def build_line_metadata(turns):
    return [
        {"index": i, "speaker": turn.get("speaker", "")}
        for i, turn in enumerate(turns)
    ]


def convert(patients, transcripts_dir, example_id_start=0):
    records = []
    example_id = example_id_start
    for orig_line_id, patient in enumerate(patients):
        transcript_id = patient["transcript"]
        transcript_path = os.path.join(transcripts_dir, transcript_id + ".json")
        with open(transcript_path) as f:
            turns = json.load(f)

        selected_turns = [t for t in turns if turn_in_segments(t, patient.get("segments"))]

        records.append({
            "input": build_input(selected_turns),
            "reference": patient.get("human_summary", ""),
            "example_id": example_id,
            "section": patient.get("label", ""),
            "orig_line_id": orig_line_id,
            "input_line_metadata": build_line_metadata(selected_turns),
        })
        example_id += 1
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients_file", type=str, required=True,
                         help="Path to the JSON file with a top-level 'patients' list")
    parser.add_argument("--transcripts_dir", type=str, required=True,
                         help="Directory containing '<transcript_id>.json' turn files")
    parser.add_argument("--output_file", type=str, required=True,
                         help="Path to write the converted DocLens-format JSON")
    args = parser.parse_args()

    with open(args.patients_file) as f:
        patients = json.load(f)["patients"]

    records = convert(patients, args.transcripts_dir)

    with open(args.output_file, "w") as f:
        json.dump(records, f, indent=4)

    print("Wrote %d records to %s" % (len(records), args.output_file))
