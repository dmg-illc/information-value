import pickle
from transformers import AutoTokenizer, BlenderbotTokenizer


if __name__ == "__main__":
    # Path to save the txt files
    output_path = "/../information-value/data/corpora/switchboard"

    # Prepare for different formattings
    dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    blender_tokenizer = BlenderbotTokenizer.from_pretrained(
        "facebook/blenderbot-400M-distill"
    )

    split_tokens_map = [
        ("tab", "\t"),
        ("newline", "\n"),
        ("4spaces", "    "),
        ("EOS_gpt", dialogpt_tokenizer.eos_token),
        ("EOS", blender_tokenizer.eos_token),
        (
            "EOS_BOS",
            f"{blender_tokenizer.eos_token} {blender_tokenizer.bos_token}",
        ),
    ]

    for data_split in ["train", "validation", "test"]:
        with open(
            f"/../information-value/data/corpora/switchboard/wts_{data_split}_nxt.pkl",
            "rb",
        ) as f:
            dataset = pickle.load(f)

        for split_token_name, split_token in split_tokens_map:
            with open(
                f"{output_path}/{data_split}/switchboard_{data_split}_{split_token_name}.txt",
                "w",
            ) as f:
                for instance in dataset:
                    if split_token_name == "EOS_BOS":
                        f.write(
                            split_token.join(map(str.strip, instance))
                            + blender_tokenizer.eos_token
                            + "\n"
                        )
                    else:
                        f.write(
                            split_token.join(map(str.strip, instance))
                            + split_token
                            + "\n"
                        )
