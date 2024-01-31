import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_jsonl

import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    OPTForCausalLM,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sentence_surprisal(
    context, sentence, model, max_seq_len, tokenizer, separator=None
):
    """
    Process sentence and return the negative log probability of each token

    Special token formats for in/out of context samples
    - in_context: `<sep/bos> context <sep/none> target`
    - oo_context: `<sep/bos> target`
    """
    context_ids = tokenizer(context)["input_ids"]
    sentence_ids = tokenizer(sentence)["input_ids"]

    # Input: Join context and sentence
    if separator is not None:
        input_ids = [separator] + context_ids + [separator] + sentence_ids
    else:
        input_ids = [tokenizer.bos_token_id] + context_ids + sentence_ids
    # Cut the input to the maximum length of the model
    input_ids = input_ids[-max_seq_len:]
    input_ids = torch.tensor(input_ids, device=model.device)

    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        log_probs = torch.nn.functional.log_softmax(outputs.logits.squeeze(0), dim=-1)
        # Get the log probability for each token in the sentence (index from the end of input_ids, not the start of context)
        #         in_context_token_log_probs = log_probs[range(len(context_ids), len(input_ids) - 1), sentence_ids]
        in_context_token_log_probs = log_probs[
            range(len(input_ids) - len(sentence_ids) - 1, len(input_ids) - 1),
            sentence_ids,
        ]
        # Get the conditional entropy for each token in the sentence
        probs = torch.exp(log_probs)
        in_context_entropies = -(log_probs * probs).nansum(-1)
        # Get the deviation of the information content from the conditional entropy
        in_context_deviations = torch.abs(
            (-in_context_token_log_probs)
            -
            #             in_context_entropies[range(len(context_ids), len(input_ids) - 1)]
            in_context_entropies[
                range(len(input_ids) - len(sentence_ids) - 1, len(input_ids) - 1)
            ]
        )

    # Now out of context
    assert len(sentence_ids) < max_seq_len, "Sentence too long"
    if separator is not None:
        input_ids = torch.tensor([separator] + sentence_ids, device=model.device)
    else:
        input_ids = torch.tensor(
            # tokenizer(' ')["input_ids"] + sentence_ids, device=model.device
            [tokenizer.bos_token_id] + sentence_ids, device=model.device
        )
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        log_probs = torch.nn.functional.log_softmax(outputs.logits.squeeze(0), dim=-1)
        out_of_context_token_log_probs = log_probs[
            range(len(input_ids) - 1), sentence_ids
        ]
        probs = torch.exp(log_probs)
        out_of_context_entropies = -(log_probs * probs).nansum(-1)
        out_of_context_deviations = torch.abs(
            (-out_of_context_token_log_probs)
            - out_of_context_entropies[range(len(input_ids) - 1)]
        )

    rdict = {
        "in_context_surprisal": -in_context_token_log_probs,
        "out_of_context_surprisal": -out_of_context_token_log_probs,
        "in_context_entropies": in_context_entropies,
        "out_of_context_entropies": out_of_context_entropies,
        "in_context_deviations": in_context_deviations,
        "out_of_context_deviations": out_of_context_deviations,
    }
    for k in rdict:
        rdict[k] = rdict[k].numpy()

    return rdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument(
        "--model_name", type=str, help="Name of the huggingface model to be used."
    )
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--separator", type=str, default=None)
    parser.add_argument("--debug_instances", type=int, default=None)
    parser.add_argument("--logging", action="store_true", default=False)
    parser.add_argument("--random_contexts", action="store_true", default=True)

    args = parser.parse_args()

    # Load model and tokenizer
    print(args)
    print("Initializing model and loading data...")
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )
    if "facebook/opt" in args.model_name:
        model = OPTForCausalLM.from_pretrained(
            args.model_name,
        )  # device_map="auto")
    else:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name,
        )  # device_map="auto")

    # Deal with HF config differences
    if "n_positions" in model.config.__dict__.keys():
        max_seq_len = model.config.n_positions
    else:
        max_seq_len = model.config.max_position_embeddings

    # Load corpus
    data = load_jsonl(args.corpus_path)

    # Collect results
    results = defaultdict(list)

    print(f"Dataset size: {len(data)}")
    if args.debug_instances is not None:
        n_rows = args.debug_instances
        print(f"Debugging on {n_rows} instances")
    else:
        n_rows = len(data)

    if (
        "switchboard" in args.corpus_path
        or "dailydialog" in args.corpus_path
        or "BLL2018" in args.corpus_path
    ):
        contexts = {elem["context_id"]: elem["context"] for elem in data}
    else:
        contexts = {elem["id"]: elem["context"] for elem in data}
    all_context_ids = list(contexts.keys())
 
    # Collect list of context ids with empty contexts (only for RT/Clasp datasets)
    if "BLL2018" in args.corpus_path:
        empty_context_ids = {d['context_id'] for d in data if len(d['context']) == 0}
    elif "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
        empty_context_ids = set()
    else:
        empty_context_ids = {c for c in all_context_ids if '_0' in c}
    print(f'Proportion of empty contexts: {(len(empty_context_ids) / len(data)) * 100:.2f}%')

    # Loop through stimuli ((context, response) pairs)
    for i, row in tqdm(enumerate(data[:n_rows]), total=n_rows):

        # Save acceptability or RT judgements
        judgements = dict()
        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            judgements["judgements"] = row["judgements"]
            judgements["mean_acceptability"] = np.mean(row["judgements"])
            judgements["median_acceptability"] = np.median(row["judgements"])
        if "BLL2018" in args.corpus_path:
            judgements["judgements_in_context"] = row["judgements_in_context"]
            judgements["mean_acceptability_in_context"] = np.mean(
                row["judgements_in_context"]
            )
            judgements["median_acceptability_in_context"] = np.median(
                row["judgements_in_context"]
            )
            judgements["judgements_out_of_context"] = row["judgements_out_of_context"]
            judgements["mean_acceptability_out_of_context"] = np.mean(
                row["judgements_out_of_context"]
            )
            judgements["median_acceptability_out_of_context"] = np.median(
                row["judgements_out_of_context"]
            )
        elif "_rt" in args.corpus_path:
            for k in row.keys():
                if k not in [
                    "id",
                    "text_id_",
                    "sentence_num_",
                    "judgements",
                    "context",
                    "target",
                ]:
                    judgements[k] = row[k]

        if "_rt" in args.corpus_path:
            context_id = row["id"]
        else:
            context_id = row["context_id"]
        results["context_id"].append(context_id)

        # Select a random (non-overlapping, non-empty) context 
        if args.random_contexts:
            choose_context_ids = set(contexts.keys()).copy() 
            choose_context_ids.remove(context_id)
            choose_context_ids = choose_context_ids - empty_context_ids
            rnd_context_id = random.choice(list(choose_context_ids))
            results["random_context_id"].append(rnd_context_id)

        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            results["target_id"].append(row["target_id"])
        elif "BLL2018" in args.corpus_path:
            results["target_id"].append(row["language"])
        else:
            pass

        for k, v in judgements.items():
            results[k].append(v)

        if "switchboard" in args.corpus_path or "dailydialog" in args.corpus_path:
            results["real"].append("True" if int(row["real"]) else "False")
        elif "BLL2018" in args.corpus_path:
            results["real"].append("True" if row["language"] == "English" else "False")

        # Compute surprisal and add to results dict
        surprisal_dict = get_sentence_surprisal(
            row["context"], row["target"], model, max_seq_len, tokenizer
        )
        for k, v in surprisal_dict.items():
            results[k].append(v)

        # Compute surprisal with random context and add to results dict
        if args.random_contexts:
            rand_surprisal_dict = get_sentence_surprisal(
                contexts[rnd_context_id], row["target"], model, max_seq_len, tokenizer
            )
            for k, v in rand_surprisal_dict.items():
                results[f"{k}_rnd"].append(v)

        _time_start = time.time()

    if args.logging:
        logger.warning(
            f"Elapsed time (s) for surprisal metric calculations: {time.time() - _time_start}"
        )

    results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results.to_csv(f"{args.output_path}.csv", index=False)
