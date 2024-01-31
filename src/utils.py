import json
import os
from types import SimpleNamespace
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import random
import numpy


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def load_jsonl(path: str, return_obj=False) -> List[Dict[str, Any]]:
    data = []
    # open with utf8 encoding to avoid UnicodeDecodeError
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data.append(
                json.loads(line, object_hook=lambda d: SimpleNamespace(**d))
                if return_obj
                else json.loads(line)
            )
    return data


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_information_value_data(data_dir: str, columns: List[str] = None):
    """
    Load data from a directory of CSV files containing surprisal estimates.
    # Arguments:
        data_dir: Path to directory containing CSV files.
        columns: List of columns to load from CSV files. If None, all columns are loaded.
    """
    data = None
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            fields_from_filename = parse_information_value_filename(file)
            if columns is None:
                df = pd.read_csv(os.path.join(data_dir, file))
            else:
                df = pd.read_csv(os.path.join(data_dir, file), usecols=columns)

            for field in fields_from_filename:
                df[field] = fields_from_filename[field]

            if "data" in locals():
                data = pd.concat([data, df])
            else:
                data = df
    print(f"Size of dataset: {len(data)} rows")
    return data


def parse_information_value_filename(filename: str):
    """
    Parse a filename of the form into a dictionary of fields.
    # Arguments:
        filename: Filename to parse.
    """
    corpus, model, sampling_str, _, _, _ = filename.split("-")
    if len(sampling_str.split("_")) == 1:
        sampling = sampling_str
        sampling_param = "None"
    else:
        sampling, sampling_param = sampling_str.split("_")
    return {
        "corpus": corpus,
        "model": model,
        "sampling": sampling,
        "sampling_param": sampling_param
    }