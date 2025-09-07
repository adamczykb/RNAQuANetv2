import logging
import os
import tarfile

import pandas as pd
import requests
from tqdm import tqdm


def prepare_raw_data(url: str, name: str) -> dict[str, bool]:
    """Download, extract and clean old data for further processing.

    Args:
        url: tar.gz dataset url
        name: dataset name
    Returns:
        tuple: (train_exists, val_exists, test_exists)
    """
    if not os.path.exists(f"data/01_raw/{name}"):
        os.makedirs(f"data/01_raw/{name}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
            with open(f"data/01_raw/{name}/archive.tar.gz", "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        with tarfile.open("data/01_raw/{name}/archive.tar.gz", "r:gz") as tar:
            tar.extractall(f"data/01_raw/{name}")

    return dict(
        {
            "train_csv_exists": os.path.exists(f"data/01_raw/{name}/train.csv"),
            "validate_csv_exists": os.path.exists(f"data/01_raw/{name}/val.csv"),
            "test_csv_exists": os.path.exists(f"data/01_raw/{name}/test.csv"),
        }
    )
