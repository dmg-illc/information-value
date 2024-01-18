import argparse
import os
import random
import time
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from utils import load_jsonl
from scorer import Scorer

EOS_SEPARATOR = "</s>"
spcy = spacy.load("en_core_web_sm")

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_distances(
    alternatives: List[str],
    tgt_sentence: str,
    scorer: Scorer,
    max_alternatives=None,
    max_len=1000000,
    logging=False,
):
    scores = defaultdict(list)

    if max_alternatives:
        alternatives = alternatives[:max_alternatives]

    # Lexical and syntactic distances
    # n-gram order
    for n in range(1, 4):
        # tokens and pos n-grams
        for pos_bool, pos_str in [(False, ""), (True, "_pos")]:

            time_start = time.time()
            # Compute the distance with the target
            for candidate in alternatives:
                score = scorer.ngram_overlap(
                    tgt_sentence,
                    candidate,
                    max_len1=max_len,
                    max_len2=max_len,
                    n=n,
                    pos=pos_bool,
                )
                scores[f"{n}gram{pos_str}_cross"].append(
                    1 - score
                )  # transform overlap to distance
            if logging:
                logger.warning(
                    f"Time elapsed for cross {n}-gram{pos_str}-overlap : {time.time() - time_start} seconds"
                )

            # Compute the distance with each alternative
            time_start = time.time()
            for alt1_idx, alt1 in enumerate(alternatives):
                for alt2_idx, alt2 in enumerate(alternatives):
                    if alt1_idx < alt2_idx:
                        score = scorer.ngram_overlap(
                            alt1,
                            alt2,
                            max_len1=max_len,
                            max_len2=max_len,
                            n=n,
                            pos=pos_bool,
                        )
                        scores[f"{n}gram{pos_str}_self"].append(1 - score)
            if logging:
                logger.warning(
                    f"Time elapsed for self {n}-gram{pos_str}-overlap : {time.time() - time_start} seconds"
                )

    # Semantic distance
    # first obtain embeddings for target and candidate responses
    time_start = time.time()
    all_embeddings = scorer.compute_embeddings(
        [tgt_sentence] + alternatives, max_len=max_len
    )
    if logging:
        logger.warning(
            f"Time elapsed for computing embeddings: {time.time() - time_start} seconds"
        )
    tgt_embedding = all_embeddings[0, :]
    cand_embeddings = all_embeddings[1:, :]

    # then compute cosine and euclidean similarity
    time_start = time.time()
    for score_name, score_func in [
        ("cosine", scorer.cosine_distance),
        ("euclidean", scorer.euclidean_distance),
    ]:
        for cand_idx in range(len(alternatives)):
            score = score_func(cand_embeddings[cand_idx, :], tgt_embedding)
            scores[f"{score_name}_cross"].append(score)
        for alt1_idx, alt1 in enumerate(alternatives):
            for alt2_idx, alt2 in enumerate(alternatives):
                if alt1_idx < alt2_idx:
                    score = score_func(
                        cand_embeddings[alt1_idx, :], cand_embeddings[alt2_idx, :]
                    )
                    scores[f"{score_name}_self"].append(score)
    if logging:
        logger.warning(
            f"Time elapsed for computing semantic distances: {time.time() - time_start} seconds"
        )

    # Absolute length difference
    time_start = time.time()
    tgt_length = scorer.length(tgt_sentence)
    for candidate in alternatives:
        scores["length_cross"].append(abs(tgt_length - scorer.length(candidate)))
    for alt1_idx, alt1 in enumerate(alternatives):
        for alt2_idx, alt2 in enumerate(alternatives):
            if alt1_idx < alt2_idx:
                scores["length_self"].append(
                    abs(scorer.length(alt1) - scorer.length(alt2))
                )
    if logging:
        logger.warning(
            f"Time elapsed for computing length differences: {time.time() - time_start} seconds"
        )
    return scores


def get_distances_fast(
    alternatives: List[str],
    tgt_sentence: str,
    scorer: Scorer,
    max_alternatives=None,
    max_len=1000000,
    logging=False
):
    MAX_NGRAM_ORDER = 3
    scores = defaultdict(list)

    if max_alternatives:
        alternatives = alternatives[:max_alternatives]

    # Lexical and syntactic distances (and unigram length difference)
    time_start = time.time()
    # Compute the distance with the target
    ngram_distances = scorer.ngram_distance(
        alternatives, tgt_sentence, max_n=MAX_NGRAM_ORDER, return_length_diff=True
    )

    for n in range(1, MAX_NGRAM_ORDER + 1):
        for pos_bool, pos_str in [(False, ""), (True, "_pos")]:
            scores[f"{n}gram{pos_str}_cross"].extend(
                ngram_distances["cross"][n][pos_bool]
            )
            scores[f"{n}gram{pos_str}_self"].extend(
                ngram_distances["self"][n][pos_bool]
            )
    scores["length_cross"].extend(ngram_distances["cross"]["length"])
    scores["length_self"].extend(ngram_distances["self"]["length"])
    if logging:
        logger.warning(
            f"Time elapsed for overlap and length calculations : {time.time() - time_start} seconds"
        )

    # Semantic distance
    # first obtain embeddings for target and candidate responses
    time_start = time.time()
    all_embeddings = scorer.compute_embeddings(
        [tgt_sentence] + alternatives, max_len=max_len
    )
    if logging:
        logger.warning(
            f"Time elapsed for computing embeddings: {time.time() - time_start} seconds"
        )
    tgt_embedding = all_embeddings[0, :]
    cand_embeddings = all_embeddings[1:, :]

    # then compute cosine and euclidean similarity
    time_start = time.time()
    for score_name, score_func in [
        ("cosine", scorer.cosine_distance),
        ("euclidean", scorer.euclidean_distance),
    ]:
        for cand_idx in range(len(alternatives)):
            score = score_func(cand_embeddings[cand_idx, :], tgt_embedding)
            scores[f"{score_name}_cross"].append(score)
        for alt1_idx, alt1 in enumerate(alternatives):
            for alt2_idx, alt2 in enumerate(alternatives):
                if alt1_idx < alt2_idx:
                    score = score_func(
                        cand_embeddings[alt1_idx, :], cand_embeddings[alt2_idx, :]
                    )
                    scores[f"{score_name}_self"].append(score)
    if logging:
        logger.warning(
            f"Time elapsed for computing semantic distances: {time.time() - time_start} seconds"
        )

    return scores


def distances_to_surprise(distances, aggregation_func):
    return {k: aggregation_func(v) for k, v in distances.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--alternatives_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--separator", type=str, default=None)
    parser.add_argument("--keep_separator", action="store_true", default=False)
    parser.add_argument("--debug_instances", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--step_size_samples", type=int, default=10)
    parser.add_argument("--logging", action="store_true", default=False)
    parser.add_argument("--random_contexts", action="store_true", default=False)

    args = parser.parse_args()

    # Load corpus
    data = load_jsonl(args.corpus_path)

    # Load generated alternatives
    _time_start = time.time()
    alternatives = defaultdict(list)
    for row in load_jsonl(args.alternatives_path):
        for i, alternative in enumerate(row["alternatives"]):
            if i >= args.max_samples:
                break
            alternative = alternative.strip()
            if args.separator and alternative != "":
                if args.separator == "spacy":
                    doc = spcy(alternative)
                    alternative = doc.sents.__next__().text
                else:
                    alternative = (
                        alternative.split(args.separator)[0] + args.separator
                        if args.keep_separator
                        else alternative.split(args.separator)[0]
                    )
            alternatives[row["context_id"]].append(alternative)
    all_context_ids = list(alternatives.keys())

    if args.logging:
        logger.warning(
            f"Elapsed time (s) for loading alternatives: {time.time() - _time_start}"
        )

    results = defaultdict(list)
    scorer = Scorer()

    n_rows = args.debug_instances if args.debug_instances else len(data)
    for i, row in tqdm(enumerate(data), total=n_rows):
        if args.debug_instances and i >= args.debug_instances:
            break

        judgements = dict()
        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            judgements["judgements"] = row["judgements"]
            judgements["mean_acceptability"] = np.mean(row["judgements"])
            judgements["median_acceptability"] = np.median(row["judgements"])
        if "BLL2018" in args.corpus_path:
            judgements["judgements_in_context"] = row["judgements_in_context"]
            judgements["mean_acceptability_in_context"] = np.mean(row["judgements_in_context"])
            judgements["median_acceptability_in_context"] = np.median(row["judgements_in_context"])
            judgements["judgements_out_of_context"] = row["judgements_out_of_context"]
            judgements["mean_acceptability_out_of_context"] = np.mean(row["judgements_out_of_context"])
            judgements["median_acceptability_out_of_context"] = np.median(row["judgements_out_of_context"])
        elif "_rt" in args.corpus_path:
            for k in row.keys():
                if k not in ["id", "text_id_", "sentence_num_", "judgements", "context", "target"]:
                    judgements[k] = row[k]

        if "_rt" in args.corpus_path:
            context_id = row["id"]
        else:
            context_id = row["context_id"]

        results["context_id"].extend(
            (args.max_samples // args.step_size_samples) * [context_id]
        )

        if args.random_contexts:
            all_context_ids_tmp = all_context_ids.copy()
            all_context_ids_tmp.remove(context_id)
            all_context_ids_tmp.remove("<no-context>")
            rnd_context_id = random.choice(all_context_ids_tmp)
            results["random_context_id"].extend(
                (args.max_samples // args.step_size_samples) * [rnd_context_id]
            )

        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            results["target_id"].extend(
                (args.max_samples // args.step_size_samples) * [row["target_id"]]
            )
        elif "BLL2018" in args.corpus_path:
            results["target_id"].extend(
                (args.max_samples // args.step_size_samples) * [row["language"]]
            )
        else:
            pass  # context id contains target id

        for k, v in judgements.items():
            results[k].extend(
                (args.max_samples // args.step_size_samples) * [v]
            )

        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            results["real"].extend(
                (args.max_samples // args.step_size_samples) * ["True"]
                if int(row["real"])
                else (args.max_samples // args.step_size_samples) * ["False"]
            )
        elif "BLL2018" in args.corpus_path:
            results["real"].extend(
                (args.max_samples // args.step_size_samples) * ["True"]
                if row["language"] == "English"
                else (args.max_samples // args.step_size_samples) * ["False"]
            )

        distances_in_context = get_distances_fast(
            alternatives[context_id], row["target"], scorer, logging=args.logging
        )
        distances_out_of_context = get_distances_fast(
            alternatives["<no-context>"], row["target"], scorer, logging=args.logging
        )
        if args.random_contexts:
            distances_in_random_context = get_distances_fast(
                alternatives[rnd_context_id], row["target"], scorer, logging=args.logging
            )
        _time_start = time.time()
        for n in range(
            args.step_size_samples, args.max_samples + 1, args.step_size_samples
        ):
            results["n_samples"].append(n)
            for k, v in sorted(distances_in_context.items()):
                _v = v[:n]
                if "_cross" in k:
                    score_name = k[: -len("_cross")]
                    results[f"surprise_mean_{score_name}"].append(np.mean(_v))
                    results[f"surprise_min_{score_name}"].append(np.min(_v))
                    results[f"surprise_max_{score_name}"].append(np.max(_v))
                if "_self" in k:
                    score_name = k[: -len("_self")]
                    results[f"expected_surprise_{score_name}"].append(np.mean(_v))
                    results[f"surprise_deviation_{score_name}"].append(
                        abs(
                            results[f"surprise_mean_{score_name}"][-1]
                            - results[f"expected_surprise_{score_name}"][-1]
                        )
                    )

            for k, v in sorted(distances_out_of_context.items()):
                _v = v[:n]
                if "_cross" in k:
                    score_name = k[: -len("_cross")]
                    results[f"surprise_ooc_mean_{score_name}"].append(np.mean(_v))
                    results[f"surprise_ooc_min_{score_name}"].append(np.min(_v))
                    results[f"surprise_ooc_max_{score_name}"].append(np.max(_v))
                    results[f"context_informativeness_{score_name}"].append(
                        results[f"surprise_ooc_mean_{score_name}"][-1]
                        - np.mean(distances_in_context[k])
                    )
                if "_self" in k:
                    score_name = k[: -len("_self")]
                    results[f"expected_surprise_ooc_{score_name}"].append(np.mean(_v))
                    results[f"expected_context_informativeness_{score_name}"].append(
                        results[f"expected_surprise_ooc_{score_name}"][-1]
                        - np.mean(distances_in_context[k])
                    )
                    results[f"surprise_ooc_deviation_{score_name}"].append(
                        abs(
                            results[f"surprise_ooc_mean_{score_name}"][-1]
                            - results[f"expected_surprise_ooc_{score_name}"][-1]
                        )
                    )

            if args.random_contexts:
                for k, v in sorted(distances_in_random_context.items()):
                    _v = v[:n]
                    if "_cross" in k:
                        score_name = k[: -len("_cross")]
                        results[f"surprise_rnd_mean_{score_name}"].append(np.mean(_v))
                        results[f"surprise_rnd_min_{score_name}"].append(np.min(_v))
                        results[f"surprise_rnd_max_{score_name}"].append(np.max(_v))
                    if "_self" in k:
                        score_name = k[: -len("_self")]
                        results[f"expected_rnd_surprise_{score_name}"].append(np.mean(_v))
                        results[f"surprise_rnd_deviation_{score_name}"].append(
                            abs(
                                results[f"surprise_rnd_mean_{score_name}"][-1]
                                - results[f"expected_rnd_surprise_{score_name}"][-1]
                            )
                        )
        if args.logging:
            logger.warning(
                f"Elapsed time (s) for surprisal metric calculations: {time.time() - _time_start}"
            )

    results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results.to_csv(args.output_path, index=False)
