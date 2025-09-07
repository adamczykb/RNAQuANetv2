import json
import logging
import os
import re
import shutil
import tempfile
import traceback
from typing import List, Literal
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser
from rnapolis import parser
from rnapolis.common import BasePair, LeontisWesthof, Residue, ResidueAuth

from pipelines.structure_processing.descriptive_columns import *
from pipelines.structure_processing.feature_compression import (
    compress_adeine_features,
    compress_cytosine_features,
    compress_guanine_features,
    compress_uracil_features,
)

COMMON_COLUMNS = (
    ["atr_" + i for i in ATR_COMMON_COLUMNS]
    + ["bon_" + i for i in BON_COMMON_COLUMNS]
    + ["aag_" + i for i in AAG_COMMON_COLUMNS]
)


def extract_nucleotides(
    filtered_pdb_file_path: str, atom_for_distance_calculations: str
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    pdb_parser = PDBParser(QUIET=True)
    sequence = []
    x = []
    y = []
    z = []
    structure_pdb = pdb_parser.get_structure("structure", filtered_pdb_file_path)
    try:
        for model in structure_pdb:
            for chain in model:
                for residue in chain:
                    sequence.append(re.sub(r"[^ATGCU]", "", residue.resname))
                    x.append(residue[atom_for_distance_calculations].get_coord()[0])
                    y.append(residue[atom_for_distance_calculations].get_coord()[1])
                    z.append(residue[atom_for_distance_calculations].get_coord()[2])
        df = pd.DataFrame(
            {"nucleotide": sequence, "x": x, "y": y, "z": z},
        )
        return df
    except Exception:
        logger.error(
            "atom_coordinates " + filtered_pdb_file_path + " " + traceback.format_exc()
        )
        raise Exception("atom_coordinates")


def fred_basepairs(pdb_path: str, fr3d_url: str):
    logger = logging.getLogger(__name__)
    basepairs = []
    try:
        files = {"file": open(pdb_path, "rb")}
        response = requests.post(f"{fr3d_url}/pdb", files=files, timeout=10)
        if response.status_code == 200:
            basepairs = json.loads(response.text)
        else:
            logger.error("fr3d_processing" + traceback.format_exc())
            raise Exception("fr3d")
    except Exception:
        logger.error("fr3d_processing" + response.text + traceback.format_exc())
        raise Exception("fr3d")
    basepairs_rnapolis = []
    basepairs_features = []

    for chain in basepairs:
        for bp in chain["annotations"]:
            nt1 = Residue(
                None,
                ResidueAuth(
                    bp["chain1"],
                    int(bp["3d_id1"]),
                    bp["icode1"] if bp["icode1"] != "?" else None,
                    bp["nt1"],
                ),
            )
            nt2 = Residue(
                None,
                ResidueAuth(
                    bp["chain2"],
                    int(bp["3d_id2"]),
                    bp["icode2"] if bp["icode2"] != "?" else None,
                    bp["nt2"],
                ),
            )
            lw = LeontisWesthof(bp["bp"])
            saenger = None  # detect_saenger(nt1, nt2, lw)
            basepairs_rnapolis.append(BasePair(nt1, nt2, lw, saenger))
            basepairs_features.append([])
    return zip(basepairs_rnapolis, basepairs_features)


def x3dna_basepairs(pdb_path: str, x3dna_path: str):
    logger = logging.getLogger(__name__)
    basepairs = []
    if os.path.exists(f"/home/adamczykb/tmp/{pdb_path}.json"):
        try:
            with open(f"/home/adamczykb/tmp/{pdb_path}.json", "r") as f:
                basepairs = json.loads(f.read())
        except Exception:
            logger.error(
                "x3dna_processing " + f"/tmp/{pdb_path}.json" + traceback.format_exc()
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.copy(pdb_path, temp_dir)
                logger.info("x3dna_processing " + pdb_path)
                process = Popen(
                    [
                        x3dna_path,
                        "more",
                        f"-i={temp_dir}/{pdb_path.split('/')[-1]}",
                        "json",
                    ],
                    stdout=PIPE,
                    stderr=PIPE,
                    cwd=temp_dir,
                )
                stdout1, stderr = process.communicate()
                if stderr is not None and stdout1 is None:
                    logger.error("x3dna_processing " + stderr.decode())
                    raise Exception(stderr.decode())
                process2 = Popen(
                    ["jq", ".pairs"],
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=PIPE,
                )
                stdout2, stderr2 = process2.communicate(input=stdout1)
                if stderr2:
                    logger.error("x3dna_processing " + stderr2.decode())
                    raise Exception(stderr2.decode())
                basepairs = json.loads(stdout2.decode())
                os.makedirs(
                    "/".join(f"/home/adamczykb/tmp/{pdb_path}.json".split("/")[:-1]),
                    exist_ok=True,
                )
                with open(f"/home/adamczykb/tmp/{pdb_path}.json", "wb") as f:
                    f.write(stdout2)
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.copy(pdb_path, temp_dir)
                # logger.info("x3dna_processing " + pdb_path)
                process = Popen(
                    [
                        x3dna_path,
                        "more",
                        f"-i={temp_dir}/{pdb_path.split('/')[-1]}",
                        "json",
                    ],
                    stdout=PIPE,
                    stderr=PIPE,
                    cwd=temp_dir,
                )
                stdout1, stderr = process.communicate()
                if stderr is not None and stdout1 is None:
                    logger.error("x3dna_processing " + stderr.decode())
                    raise Exception(stderr.decode())
                process2 = Popen(
                    ["jq", ".pairs"],
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=PIPE,
                )
                stdout2, stderr2 = process2.communicate(input=stdout1)
                if stderr2:
                    logger.error("x3dna_processing " + stderr2.decode())
                    raise Exception(stderr2.decode())
                basepairs = json.loads(stdout2.decode())
                os.makedirs(
                    "/".join(f"/home/adamczykb/tmp/{pdb_path}.json".split("/")[:-1]),
                    exist_ok=True,
                )
                with open(f"/home/adamczykb/tmp/{pdb_path}.json", "w") as f:
                    f.write(stdout2.decode())
        except Exception:
            logger.error("x3dna_processing " + pdb_path + " " + traceback.format_exc())
            # raise Exception("x3dna")
            return zip([], [])
    basepairs_rnapolis = []
    basepairs_features = []
    if type(basepairs) == list:
        for bp in basepairs:
            name_num = bp["nt1"].split("^")[0].split(".")[1]
            name = ""
            cut = 0
            for name_char_ind in range(len(name_num)):
                if name_num[name_char_ind].isalpha():
                    name += name_num[name_char_ind]
                else:
                    cut = name_char_ind
                    break
            nt1 = Residue(
                None,
                ResidueAuth(
                    bp["nt1"].split(".")[0],
                    int(name_num[cut:]),
                    bp["nt1"].split("^")[1] if len(bp["nt1"].split("^")) == 2 else None,
                    name,
                ),
            )
            name_num = bp["nt2"].split("^")[0].split(".")[1]
            name = ""
            cut = 0
            for name_char_ind in range(len(name_num)):
                if name_num[name_char_ind].isalpha():
                    name += name_num[name_char_ind]
                else:
                    cut = name_char_ind
                    break
            nt2 = Residue(
                None,
                ResidueAuth(
                    bp["nt2"].split(".")[0],
                    int(name_num[cut:]),
                    bp["nt2"].split("^")[1] if len(bp["nt2"].split("^")) == 2 else None,
                    name,
                ),
            )
            lw = "bp_uncertain" if "." in bp["LW"] else bp["LW"]
            saenger = None  # detect_saenger(nt1, nt2, lw)
            basepairs_rnapolis.append(BasePair(nt1, nt2, lw, saenger))
            basepairs_features.append(
                {
                    "C1C1_dist": bp["C1C1_dist"],
                    "N1N9_dist": bp["N1N9_dist"],
                    "C6C8_dist": bp["C6C8_dist"],
                    "hbonds_num": bp["hbonds_num"],
                    "interBase_angle": bp["interBase_angle"],
                    "planarity": bp["planarity"],
                    "simple_Shear": bp["simple_Shear"],
                    "simple_Stretch": bp["simple_Stretch"],
                    "simple_Buckle": bp["simple_Buckle"],
                    "simple_Propeller": bp["simple_Propeller"],
                    "bp_params_1": bp["bp_params"][0],
                    "bp_params_2": (
                        bp["bp_params"][1] if len(bp["bp_params"]) > 1 else 0
                    ),
                    "bp_params_3": (
                        bp["bp_params"][2] if len(bp["bp_params"]) > 2 else 0
                    ),
                    "bp_params_4": (
                        bp["bp_params"][3] if len(bp["bp_params"]) > 3 else 0
                    ),
                    "bp_params_5": (
                        bp["bp_params"][4] if len(bp["bp_params"]) > 4 else 0
                    ),
                    "bp_params_6": (
                        bp["bp_params"][5] if len(bp["bp_params"]) > 5 else 0
                    ),
                }
            )
        return zip(basepairs_rnapolis, basepairs_features)
    # logger.warning(f"No basepairs found for: {pdb_path} {basepairs}")
    return zip([], [])


def extract_pairing(filtered_pdb_file_path: str, tool_url_name: str):
    """
    Extract basepairs from filtered pdb file.

    Args:
    - filtered_pdb_file_path - absolute path to filtered pdb file
    - distinguish - True if distinguish between opening and closing parentheses during one hot encoding

    Returns:
    - Pandas dataframe with basepair node features
    """
    pairings = {bp_t: [] for bp_t in PAIRINGS_FEATURES}
    residues = []

    with open(filtered_pdb_file_path, "r") as f:
        structure3d = parser.read_3d_structure(f)
        paired_with_dict = {}
        if tool_url_name["name"] == "fr3d":
            for i in fred_basepairs(filtered_pdb_file_path, tool_url_name["path"]):
                i = i[0]
                donor = paired_with_dict.get(i.nt1.auth, [])
                donor.append(tuple([i.nt2.full_name, i.lw]))
                paired_with_dict[i.nt1.full_name] = donor

                acceptor = paired_with_dict.get(i.nt2.auth, [])
                acceptor.append(tuple([i.nt1.full_name, i.lw]))
                paired_with_dict[i.nt2.full_name] = acceptor
            for i in structure3d.residues:
                if i.name not in ["A", "U", "G", "C"]:
                    continue
                residues.append(str(i))
            for i in residues:
                residue_pairings = {bp_t: np.nan for bp_t in PAIRINGS_FEATURES}
                if i in paired_with_dict.keys():
                    for residue in paired_with_dict[i]:
                        residue_pairings[residue[1]] = residues.index(residue[0])
                for bp_t in PAIRINGS_FEATURES:
                    pairings[bp_t].append(residue_pairings[bp_t])
        elif tool_url_name["name"] == "x3dna":
            for i in x3dna_basepairs(filtered_pdb_file_path, tool_url_name["path"]):
                features = i[1]
                i = i[0]
                donor = paired_with_dict.get(i.nt1.auth, [])
                donor.append(tuple([i.nt2.full_name, i.lw, features]))
                paired_with_dict[i.nt1.full_name] = donor

                acceptor = paired_with_dict.get(i.nt2.auth, [])
                acceptor.append(tuple([i.nt1.full_name, i.lw, features]))
                paired_with_dict[i.nt2.full_name] = acceptor
            for i in structure3d.residues:
                if i.name not in ["A", "U", "G", "C"]:
                    continue
                residues.append(str(i))
            for i in residues:
                residue_pairings = {bp_t: np.nan for bp_t in PAIRINGS_FEATURES}
                if i in paired_with_dict.keys():
                    for residue in paired_with_dict[i]:
                        try:
                            residue_pairings[residue[1]] = [
                                residues.index(residue[0])
                            ] + list(residue[2].values())
                        except Exception as e:
                            logging.info(
                                f"Error in {filtered_pdb_file_path} {i} {residue[0]} {residue[1]} {residue[2]} "
                                + str(e)
                            )
                            os.exit(1)
                for bp_t in PAIRINGS_FEATURES:
                    pairings[bp_t].append(residue_pairings[bp_t])
    pairings["residue"] = residues
    pairings["residue_no"] = list(range(len(residues)))
    pairings_df = pd.DataFrame(pairings)

    return pairings_df


def get_features_from_file(
    feature_directory_path: str,
    file_type: Literal["bon", "ang", "atr", "aag"],
    exclude_columns: list[str] = ["Chain", "ResNum", "iCode", "Name"],
) -> pd.DataFrame:
    """
    Get features from a given file type.

    Args:
    - feature_directory_path - absolute path directory containing feature files
    - file_type - type of file with sample's features
    (supported: bon, ang, atr)
    - nan_replacement - value representing a replacement for NaN values
    found
    - exclude_columns - list of columns which should be excluded

    Returns:
    - Pandas dataframe with features from chosen sample and file type
    """

    df = pd.read_csv(
        os.path.join(
            feature_directory_path,
            f"{os.path.splitext(os.path.basename(feature_directory_path))[0]}.{file_type}",
        ),
        sep=r"\s+",
    )
    pd.set_option("future.no_silent_downcasting", True)
    return (
        df.reset_index(drop=True)
        .drop(columns=exclude_columns)
        .infer_objects(copy=False)
        .replace("-", np.nan)
        .add_prefix(f"{file_type}_")
        .astype(float)
    )


def process_features(features_file_path) -> List[pd.DataFrame]:
    try:
        bon_features = get_features_from_file(features_file_path, "bon")
        ang_features = get_features_from_file(features_file_path, "ang")
        atr_features = get_features_from_file(features_file_path, "atr")
        aag_features = get_features_from_file(features_file_path, "aag")
        residue_features = pd.concat(
            [bon_features, ang_features, atr_features, aag_features],
            axis=1,
        )

        return residue_features

    except FileNotFoundError as e:
        e.add_note(
            f"File {e.filename} was not found. Perhaps you tried returning data list without prior feature extraction?"
        )
        raise


def clear_na_features(df: pd.DataFrame, features_preferences) -> pd.DataFrame:
    """
    Clear features from NaN values.

    Args:
    - df - Pandas dataframe with features

    Returns:
    - Pandas dataframe with NaN values replaced by 0
    """
    match features_preferences["na_solution"]:
        case "ENCODER":
            return pd.concat(
                [
                    compress_cytosine_features(df, features_preferences["BASE_PATH"]),
                    compress_uracil_features(df, features_preferences["BASE_PATH"]),
                    compress_guanine_features(df, features_preferences["BASE_PATH"]),
                    compress_adeine_features(df, features_preferences["BASE_PATH"]),
                ],
                axis=0,
            )
        case "NONE":
            return df.fillna(float("inf"))
