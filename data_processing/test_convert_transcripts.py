import json
import os
import tempfile
import unittest

from convert_transcripts import convert, parse_timestamp, turn_in_segments, build_input, build_line_metadata


class TestConvertTranscripts(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.transcripts_dir = self.tmpdir.name

        # Mirrors the real sample: 8 turns, all inside Mr. Y's segment window.
        self.turns_1 = [
            {"start": "00:01:27.220", "end": "00:01:30.320", "speaker": "Unknown",
             "text": "I think she's out this week.", "logprob": -0.4758688744078291},
            {"start": "00:01:31.080", "end": "00:01:32.320", "speaker": "Unknown",
             "text": "Monica is out this week.", "logprob": -0.1},
            {"start": "00:01:32.480", "end": "00:01:33.700", "speaker": "Unknown",
             "text": "I think Xerf is also out this week.", "logprob": -0.1},
            {"start": "00:01:34.260", "end": "00:01:34.440", "speaker": "Unknown",
             "text": "Right.", "logprob": -0.1},
            {"start": "00:01:34.740", "end": "00:01:37.280", "speaker": "Unknown",
             "text": "So let's move on to Jess.", "logprob": -0.1},
            {"start": "00:01:37.820", "end": "00:01:39.580", "speaker": "Unknown",
             "text": "Do you want me to talk about this patient?", "logprob": -0.1},
            {"start": "00:01:39.720", "end": "00:01:41.980", "speaker": "Unknown",
             "text": "I saw them with Dr. Zimp in clinic last week.", "logprob": -0.1},
            {"start": "00:01:42.440", "end": "00:01:43.000", "speaker": "Unknown",
             "text": "Who's that?", "logprob": -0.1},
        ]
        with open(os.path.join(self.transcripts_dir, "GU_2024_11_25_1.json"), "w") as f:
            json.dump(self.turns_1, f)

        # A second transcript with turns both inside and outside the patient's window,
        # to exercise the segment-based filtering.
        self.turns_2 = [
            {"start": "00:07:00.000", "end": "00:07:05.000", "speaker": "Doctor",
             "text": "Before this one wraps up.", "logprob": -0.1},
            {"start": "00:07:10.000", "end": "00:07:15.000", "speaker": "Dr. Lee",
             "text": "Now let's discuss Mr X.", "logprob": -0.1},
            {"start": "00:07:20.000", "end": "00:07:25.000", "speaker": "Nurse Kim",
             "text": "He's post-op day two.", "logprob": -0.1},
            {"start": "00:15:00.000", "end": "00:15:05.000", "speaker": "Doctor",
             "text": "That's outside the window.", "logprob": -0.1},
        ]
        with open(os.path.join(self.transcripts_dir, "GU_2024_11_25_2.json"), "w") as f:
            json.dump(self.turns_2, f)

        self.patients = [
            {
                "label": "Mr. Y",
                "match_status": "POSSIBLE ADD-ON",
                "list_row": None,
                "candidates": [],
                "segments": [{"start": "00:01:24.860", "end": "00:07:07.960"}],
                "notes": "Patient has history of left nephrectomy (2004) and chromophobe RCC.",
                "human_summary": "2004 - Left nephrectomy; large abdominal mass recurrence.",
                "transcript": "GU_2024_11_25_1",
                "question": "Discuss feasibility of surgical resection",
            },
            {
                "label": "MR x",
                "match_status": "CONFIRMED",
                "list_row": 2,
                "candidates": [],
                "segments": [{"start": "00:07:08.780", "end": "00:14:41.060"}],
                "notes": "Post-op follow-up.",
                "human_summary": "Post-operative discussion, doing well.",
                "transcript": "GU_2024_11_25_2",
                "question": "Perioperative discussion",
            },
        ]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_timestamp(self):
        self.assertAlmostEqual(parse_timestamp("00:01:27.220"), 87.220)
        self.assertAlmostEqual(parse_timestamp("01:00:00.000"), 3600.0)

    def test_turn_in_segments(self):
        segments = [{"start": "00:01:00.000", "end": "00:02:00.000"}]
        inside = {"start": "00:01:30.000"}
        outside = {"start": "00:03:00.000"}
        self.assertTrue(turn_in_segments(inside, segments))
        self.assertFalse(turn_in_segments(outside, segments))
        self.assertTrue(turn_in_segments(outside, None))
        self.assertTrue(turn_in_segments(outside, []))

    def test_build_input_format(self):
        turns = [
            {"speaker": "doctor", "text": " hi there "},
            {"speaker": "patient", "text": "hello"},
        ]
        self.assertEqual(build_input(turns), "[0][doctor] hi there\n[1][patient] hello")

    def test_convert_end_to_end(self):
        records = convert(self.patients, self.transcripts_dir)

        self.assertEqual(len(records), 2)

        first, second = records

        # example_id increments from 0, orig_line_id follows list order.
        self.assertEqual(first["example_id"], 0)
        self.assertEqual(first["orig_line_id"], 0)
        self.assertEqual(second["example_id"], 1)
        self.assertEqual(second["orig_line_id"], 1)

        # section comes from the patient label.
        self.assertEqual(first["section"], "Mr. Y")
        self.assertEqual(second["section"], "MR x")

        # reference is the human summary, verbatim.
        self.assertEqual(first["reference"], self.patients[0]["human_summary"])
        self.assertEqual(second["reference"], self.patients[1]["human_summary"])

        # First patient's window covers all 8 turns in transcript 1.
        self.assertEqual(first["input"].count("\n") + 1, 8)
        self.assertTrue(first["input"].startswith("[0][Unknown] I think she's out this week."))

        # Second patient's window excludes the first and last turns of transcript 2.
        self.assertEqual(
            second["input"],
            "[0][Dr. Lee] Now let's discuss Mr X.\n"
            "[1][Nurse Kim] He's post-op day two."
        )

    def test_convert_keys_match_expected_schema(self):
        records = convert(self.patients, self.transcripts_dir)
        expected_keys = {"input", "reference", "example_id", "section", "orig_line_id",
                          "input_line_metadata"}
        for record in records:
            self.assertEqual(set(record.keys()), expected_keys)

    def test_build_line_metadata(self):
        turns = [
            {"speaker": "Unknown", "text": "hi"},
            {"speaker": "Dr. Lee", "text": "hello"},
            {"speaker": "Nurse Kim", "text": "status"},
        ]
        metadata = build_line_metadata(turns)
        self.assertEqual(metadata, [
            {"index": 0, "speaker": "Unknown"},
            {"index": 1, "speaker": "Dr. Lee"},
            {"index": 2, "speaker": "Nurse Kim"},
        ])

    def test_convert_carries_line_metadata(self):
        records = convert(self.patients, self.transcripts_dir)
        second = records[1]

        # Second patient's window keeps turns 1 and 2 of transcript 2 (see test_convert_end_to_end).
        self.assertEqual(second["input_line_metadata"], [
            {"index": 0, "speaker": "Dr. Lee"},
            {"index": 1, "speaker": "Nurse Kim"},
        ])


if __name__ == "__main__":
    unittest.main()
