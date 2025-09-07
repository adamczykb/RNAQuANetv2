import logging
import multiprocessing as mp
import os
import shutil
import subprocess
from contextlib import closing
from shutil import copyfile

import tqdm


def _rnagrowth_subprocess_func(in_filename: str, child: str, features_preferences: dict):
    """
    Launches RNAgrowth tool and extracts features from PDB file.
    """
    def process():
        copyfile(
            os.path.join(features_preferences["tools_path"], "RNAgrowth", "completeAtomNames.dict"),
            os.path.join(child, "completeAtomNames.dict"),
        )
        copyfile(
            os.path.join(features_preferences["tools_path"], "RNAgrowth", "config.properties"),
            os.path.join(child, "config.properties"),
        )
        copyfile(in_filename, os.path.join(child, os.path.basename(in_filename)))

        s = subprocess.Popen(
            [
                "java",
                "-jar",
                os.path.join(features_preferences["tools_path"], "RNAgrowth", "RNAgrowth.jar"),
                f"./{os.path.basename(in_filename)}",
                features_preferences["atom_for_distance_calculations"],
                features_preferences["max_euclidean_distance"],
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=child,
        )
        s.wait()
        error = s.stderr.read().decode("utf-8")
        out = s.stdout.read().decode("utf-8")
        
        if error != "":
            raise Exception(f"RNAgrowth error: {error}")
        # if out != "":
            # logging.info(f"RNAgrowth output: {out}")
        os.remove(os.path.join(child, "completeAtomNames.dict"))
        os.remove(os.path.join(child, "config.properties"))
        os.remove(os.path.join(child, os.path.basename(in_filename)))
        os.remove(os.path.join(child, "rnagrowth.log"))
        if len(os.listdir(child))!=8:
            logging.error(f"RNAgrowth error: files not generated correctly {os.path.basename(in_filename)} {child}")
            raise Exception(f"RNAgrowth error: files not generated correctly {os.path.basename(in_filename)} {child}")
    try:
        process()
    except:
        shutil.rmtree(child)
        os.makedirs(child, exist_ok=True)
        process()

def extract_feature_single_structure(pdb_file_path, features_file_path, features_preferences: dict):
    """
    Extracts features for a single structure.

    Args:
        pdb_file_path (str): The path to the PDB file.
        feature_destination_path (str): The destination path for the extracted features.
        regenerate_features_when_exists (bool): Flag indicating whether to regenerate features if they already exist.

    Returns:
        None
    """

    if os.path.exists(features_file_path):
        if features_preferences["regenerate_features_when_exists"]:
            shutil.rmtree(features_file_path)
        else:
            return
    os.makedirs(features_file_path, exist_ok=True)

    tool_path = os.path.join(features_preferences["tools_path"], "RNAgrowth")
    if not os.path.exists(tool_path):
        raise FileNotFoundError(f"RNAgrowth tools ('{features_preferences['tools_path']}') does not exist.")

    _rnagrowth_subprocess_func(pdb_file_path, features_file_path, features_preferences)


def extract_features(file_paths: list, features_preferences: dict):
    """
    Extracts features from a list of file path in parallel.

    Args:
        file_paths (list): A list of tuples containing the paths of PDB files and the destination paths for the extracted features.
        regenerate_features_when_exists (bool): Flag indicating whether to regenerate features if they already exist.

    Returns:
        None
    """

    with closing(mp.Pool(mp.cpu_count())) as pool:
        pool.starmap(
            extract_feature_single_structure,
            tqdm.tqdm(
                [
                    (
                        pdb_file_path,
                        feature_destination_path,
                        features_preferences,
                    )
                    for pdb_file_path, feature_destination_path in file_paths
                ],
                unit="f",
                total=len(file_paths),
            ),
        )
