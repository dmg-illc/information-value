from datasets import load_dataset
from transformers import AutoTokenizer, BlenderbotTokenizer


if __name__ == "__main__":
    # Path to save the txt files
    output_path = "/Users/mario/code/surprise/data/corpora/dailydialog"

    # Load dataset
    dataset = load_dataset("daily_dialog")

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

    # Create txt files
    for data_split in dataset:
        for split_token_name, split_token in split_tokens_map:
            # Write all rows of this dataset into a file, using only the feature "dialog"
            with open(
                f"{output_path}/{data_split}/dailydialog_{data_split}_{split_token_name}.txt",
                "w",
            ) as f:
                for instance in dataset[data_split]:
                    if split_token_name == "EOS_BOS":
                        f.write(
                            split_token.join(map(str.strip, instance["dialog"]))
                            + blender_tokenizer.eos_token
                            + "\n"
                        )
                    else:
                        f.write(
                            split_token.join(map(str.strip, instance["dialog"]))
                            + split_token
                            + "\n"
                        )
