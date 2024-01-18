import argparse
import json
import os.path
import sys

import torch.cuda
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    BlenderbotTokenizer,
    OPTForCausalLM,
)

from utils import set_seeds, load_jsonl

# Prepare for different formattings
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
blender_tokenizer = BlenderbotTokenizer.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
separator_map = {
    "none": "",
    "tab": "\t",
    "newline": "\n",
    "4spaces": "    ",
    "EOS_gpt": dialogpt_tokenizer.eos_token,
    "EOS": blender_tokenizer.eos_token,
    "EOS_BOS": f"{blender_tokenizer.eos_token} {blender_tokenizer.bos_token}",
}

decoding_configs = {
    "ancestral": {"top_p": None, "typical_p": None, "temperature": None},
    "nucleus_08": {"top_p": 0.8, "typical_p": None, "temperature": None},
    "nucleus_085": {"top_p": 0.85, "typical_p": None, "temperature": None},
    "nucleus_09": {"top_p": 0.9, "typical_p": None, "temperature": None},
    "nucleus_095": {"top_p": 0.9, "typical_p": None, "temperature": None},
    "typical_02": {"top_p": None, "typical_p": 0.2, "temperature": None},
    "typical_03": {"top_p": None, "typical_p": 0.3, "temperature": None},
    "typical_085": {"top_p": None, "typical_p": 0.85, "temperature": None},
    "typical_095": {"top_p": None, "typical_p": 0.95, "temperature": None},
    "temp_075": {"top_p": None, "typical_p": None, "temperature": 0.75},
    "temp_125": {"top_p": None, "typical_p": None, "temperature": 1.25},
}


def main(args):
    print(args)
    print("Initializing model and loading data...")
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(device)

    data = load_jsonl(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "facebook/opt" in args.model_name:
        model = OPTForCausalLM.from_pretrained(args.model_name, device_map="auto")
    else:
        model = AutoModelWithLMHead.from_pretrained(args.model_name, device_map="auto")

    # Collect all unique contexts
    contexts = {}
    for row in data:
        context = row["context"]
        if "_rt" in args.data_path:
            context_id = f'{row["text_id_"]}_{row["sentence_num_"]}'
        else:
            context_id = row["context_id"]
        if context == "":
            context = (
                separator_map[args.context_separator]
                if args.context_separator != "none"
                else " "
            )
        if context_id not in contexts:
            contexts[context_id] = context
        if len(contexts) >= args.debug_instances:
            break
    # Add empty context
    contexts["<no-context>"] = [
        separator_map[args.context_separator]
        if args.context_separator != "none"
        else " "
    ]

    context_separator = separator_map[args.context_separator]

    print("Generating alternatives...")
    alternatives = defaultdict(list)

    for context_id, context in tqdm(contexts.items()):

        if type(context) is list:
            if context_separator == "EOS_BOS":
                context = context_separator.join(context) + separator_map["EOS"]
            else:
                context = context_separator.join(context) + context_separator
        else:
            context = context + context_separator

        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        try:
            context_ids = context_ids[
                :, -(model.config.n_positions - args.max_generation_length) :
            ]
        except AttributeError:
            context_ids = context_ids[
                :,
                -(model.config.max_position_embeddings - args.max_generation_length) :,
            ]

        # print memory usage
        # print(f"Context size: {context_ids.element_size() * context_ids.nelement() / 1024 ** 2} MB")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        for config_name, decoding_config in decoding_configs.items():
            decoded_alternatives = []
            for _ in range(args.n_sampling_runs):
                alternative_ids = model.generate(
                    context_ids,
                    max_new_tokens=args.max_generation_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    # top_k=args.top_k,
                    top_p=decoding_config["top_p"],
                    typical_p=decoding_config["typical_p"],
                    # num_beams=args.n_beams,
                    temperature=decoding_config["temperature"],
                    num_return_sequences=args.n_samples_per_run,
                )
                # some models return input_output, others just output
                alternative_ids = alternative_ids[:, context_ids.shape[-1] :]
                decoded_alternatives.extend(
                    [
                        tokenizer.decode(response, skip_special_tokens=True)
                        for response in alternative_ids
                    ]
                )
            alternatives[config_name].append(
                {"context_id": context_id, "alternatives": decoded_alternatives}
            )
            # Print size of the dictionary alternatives in mb and gb
            # print(f"Alternatives size: {sys.getsizeof(alternatives) / 1024 ** 2} MB")
            # print(f"Alternatives size: {sys.getsizeof(alternatives) / 1024 ** 3} GB")

    print("Writing responses to file...")
    for config_name in alternatives:
        filename = (
            f"{config_name}-n_{args.n_sampling_runs * args.n_samples_per_run}"
            f"-maxlen_{args.max_generation_length}-sep_{args.context_separator}"
        )
        write_responses(
            Path(os.path.join(args.out_dir, filename)).with_suffix(".jsonl"),
            alternatives[config_name],
        )
        # write_params(args)


#
# def write_params(args):
#     with open(Path(args.out_path).with_suffix(".info"), "w") as f:
#         json.dump(
#             {**vars(args), "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}, f
#         )


def write_responses(path, responses):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for response in responses:
            json.dump(response, f)
            f.write("\n")


if __name__ == "__main__":
    set_seeds(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to json file with multiple references.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the huggingface model to be used.",
    )
    parser.add_argument(
        "--n_sampling_runs",
        type=int,
        default=1,
        help="The number of sampling runs in which n_samples_per_run samples are generated."
        "The total number of samples is n_sampling_runs * n_samples_per_run.",
    )
    parser.add_argument(
        "--n_samples_per_run",
        type=int,
        default=10,
        help="The number of samples to generate (given one context) in a single sampling run.",
    )
    parser.add_argument(
        "--max_generation_length",
        type=int,
        default=100,
        help="The maximum number tokens that can be generated. This is one of the two stopping criteria. The other one is the EOS special token.",
    )
    parser.add_argument(
        "--context_separator",
        type=str,
        default="none",
        choices=["none", "tab", "newline", "4spaces", "EOS_gpt", "EOS_BOS"],
        help="The token to be used to separate multiple turns in the context, and to append at the end of the context.",
    )
    parser.add_argument(
        "--debug_instances",
        type=int,
        default=int(10e10),
        help="For test runs with only a few instances",
    )
    args = parser.parse_args()
    main(args)
