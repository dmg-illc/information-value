import json

import numpy as np
import pandas as pd
from pathlib import Path


def dialogue_corpus_to_jsonl(input_path: Path, output_path: Path = None):
    df = pd.read_csv(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.jsonl"

    def _split(string, sep, position):
        string = string.split(sep)
        return sep.join(string[:position]), sep.join(string[position:])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for idx, row in df.iterrows():
            context_id, target_id = _split(row["q_id"], "_", 2)
            datum = {
                "idx": idx,
                "context_id": context_id,
                "target_id": target_id,
                "real": int(row["target"]),
                "context": row["context_text"],
                "target": row["response_text"],
                "judgements": eval(row["all_score"]),
            }
            json.dump(datum, f)
            f.write("\n")


def bll2018_to_jsonl(input_path: Path, output_path: Path = None):
    df = pd.read_csv(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for idx, row in df.iterrows():
            context = row["Pre-Context"]
            post_context = row["Post-Context"]
            if type(context) == float and np.isnan(context):
                context = ""
            if type(post_context) == float and np.isnan(post_context):
                post_context = ""
            datum = {
                "idx": idx,
                "context_id": row["ID"],
                "language": row["Language"],
                "context": context,
                "target": row["Sentence"],
                "post_context": post_context,
                "judgements_in_context": eval(row["Without-Context Ratings"]),
                "judgements_out_of_context": eval(row["With-Context Ratings"]),
            }
            json.dump(datum, f)
            f.write("\n")


if __name__ == "__main__":
    # dialogue_corpus_to_jsonl(
    #     input_path=Path(
    #         "/data/psychometric/dailydialog/dailydailog_results_is.csv"
    #     )
    # )
    # dialogue_corpus_to_jsonl(
    #     input_path=Path(
    #         "/data/psychometric/switchboard/switchboard_results_is.csv"
    #     )
    # )
    bll2018_to_jsonl(
        input_path=Path("/Users/mario/code/surprise/data/psychometric/BLL2018/processed_ratings.csv")
    )
