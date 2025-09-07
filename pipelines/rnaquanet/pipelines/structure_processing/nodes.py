import logging
import os
import shutil
import copy
from typing import Dict, List, Type
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from pipelines.structure_processing.descriptive_columns import (
    EDGE_FEATURE_COLUMNS,
    EXTENDED_PAIRINGS_FEATURES,
    NODE_FEATURE_COLUMNS,
    PAIRINGS_FEATURES,
)
from pipelines.structure_processing.feature_process import (
    clear_na_features,
    extract_nucleotides,
    extract_pairing,
    process_features,
)
from pipelines.structure_processing.parallel_feature_extraction import (
    extract_feature_single_structure,
)


def extract_pairing_nucleobase(
    entry,
    tool_url_name: str,
    atom_for_distance_calculations: str,
) -> np.array:

    if os.path.exists(os.path.join(entry)):
        if os.path.join(entry).split(".")[-1] == "cif":
            parser = MMCIFParser(QUIET=True)
            io = MMCIFIO()
        elif os.path.join(entry).split(".")[-1] == "pdb":
            parser = PDBParser(QUIET=True)
            io = PDBIO()
        else:
            raise ValueError(
                "File type is not supported. Only .pdb and .cif files are supported."
            )
        used_chains = []
        try:
            structure = parser.get_structure("str", os.path.join(entry))[0]
        except:
            logging.error(f"Error parsing structure {os.path.join( entry)}")
            raise ValueError(f"File contains errors: {os.path.join( entry)}")
        for chain in structure:
            used_chains.append(chain.id)
        for chain in structure:
            if chain.id.strip() == "":
                for i in range(48):
                    if i <= 25:
                        if chr(i + 65) not in used_chains:
                            chain.id = chr(i + 65)
                            used_chains.append(chain.id)
                            break
                    else:
                        if chr(i + 97) not in used_chains:
                            chain.id = chr(i + 97)
                            used_chains.append(chain.id)
                            break
            for residue in chain:
                for atom in residue:
                    if atom.occupancy is None:
                        atom.occupancy = 1.0
        io.set_structure(structure)
        io.save(os.path.join(entry))
        temp_df = pd.concat(
            [
                extract_nucleotides(
                    os.path.join(entry),
                    atom_for_distance_calculations,
                ),
                extract_pairing(os.path.join(entry), tool_url_name),
            ],
            axis=1,
        )
        temp_df["description"] = os.path.splitext(entry.split("/")[-1])[0]
        temp_df["residue_no"] = np.linspace(
            0, len(temp_df) - 1, len(temp_df), dtype=int
        )
        return temp_df

    else:
        raise ValueError(f"File {os.path.join(entry)} does not exist")


def extract_features_from_dataset(
    file_path: str,
    rmsd: float,
    features_preferences: dict,
):

    results = extract_pairing_nucleobase(
        file_path,
        features_preferences["basepair_tool"],
        features_preferences["atom_for_distance_calculations"],
    )
    description = os.path.splitext(file_path.split("/")[-1])[0]
    structure_features = results.set_index(["description", "residue_no"], drop=False)
    extract_feature_single_structure(
        file_path, f"/tmp/RNAgrowth/{description}", features_preferences
    )
    results = process_features_from_dataset_f(file_path, f"/tmp/RNAgrowth/")

    df = pd.concat(
        [structure_features, results],
        axis=1,
    )
    df = clear_na_features(df.reset_index(drop=True), features_preferences)
    output_path = os.path.join("/tmp/output", description)
    os.makedirs(output_path, exist_ok=True)

    path_graph = prepare_graphs(
        description,
        df,
        rmsd,
        features_preferences,
        "/tmp/output",
    )
    feature_path = f"/tmp/output/{description}.csv"
    df.to_csv(feature_path, index=False)
    return path_graph, feature_path


def process_features_from_dataset_f(entry, features_path):
    entry = os.path.splitext(entry.split("/")[-1])[0]
    features_file_path = os.path.join(features_path, entry)
    temp_df = process_features(features_file_path)
    temp_df["description"] = entry
    temp_df["residue_no"] = range(0, len(temp_df))
    temp_df.set_index(["description", "residue_no"], inplace=True)
    shutil.rmtree(features_file_path, ignore_errors=True)
    return temp_df


def parse_bp_type_onehot(
    bp_type: str, entry_feature_vector: List[int], distance: float
) -> np.array:
    """
    Parse bp_type to one-hot encoding.

    Args:
    - bp_type - base pair type
    - entry_feature_vector - feature vector

    Returns:
    - entry_feature_vector - updated feature vector
    """
    if bp_type in EDGE_FEATURE_COLUMNS:
        index = EDGE_FEATURE_COLUMNS.index(bp_type)
        entry_feature_vector[index] = 1
    try:
        entry_feature_vector[EDGE_FEATURE_COLUMNS.index("distance")] = distance
    except:
        pass

    return entry_feature_vector


def prepare_graph_iter(
    description_name,
    structure,
    features_preferences: Type[Dict],
    RMSD: float,
    output_path,
):
    if len(description_name.split("/")) > 1:
        description_name = description_name.split("/")[0]
    if os.path.exists(os.path.join(output_path, f"{description_name}.pt")):
        return

    default_edge_features = [0.0] * len(EDGE_FEATURE_COLUMNS)
    default_edge_features = np.array(default_edge_features).astype(np.float32)

    nucleotide_onehot = pd.DataFrame(
        pd.get_dummies(
            pd.Categorical(
                structure["nucleotide"],
                categories=(
                    [
                        "A",
                        "U",
                        "G",
                        "C",
                    ]
                ),
            ),
            prefix="nucleotide",
        )
    ).astype(np.float32)
    nucleotide_onehot["index"] = structure.index
    nucleotide_onehot.set_index("index", inplace=True)
    structure["index"] = structure.index
    structure["structure_size"] = np.repeat(len(structure), len(structure))
    structure = pd.concat(
        [
            structure,
            nucleotide_onehot,
        ],
        axis=1,
    )

    edge_indices = []  # (source, target, )
    edge_attr = []  # (bp_type, distance)
    for ind, i in structure.iterrows():
        checked = []
        for pairing_type in PAIRINGS_FEATURES:
            if type(i[pairing_type]) != type([]) and pd.isna(i[pairing_type]):
                continue
            if features_preferences["pairing_features"] == "basic":
                j = structure.loc[structure.residue_no == i[pairing_type]].iloc[0]
                edge_indices.append(
                    (
                        int(i.residue_no),
                        int(i[pairing_type]),
                    )
                )
                edge_attr.append(
                    (
                        pairing_type,
                        abs(
                            ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                            ** 0.5
                        ),
                        (
                            abs(i.residue_no - j.residue_no)
                            if i["residue"].split(".")[0] == j["residue"].split(".")[0]
                            else 0
                        ),
                    )
                )
                checked.append(i[pairing_type])
                continue
            pair_list = str(i[pairing_type])
            if features_preferences["pairing_features"] == "extended":
                pairing = [
                    float(el)
                    for el in pair_list.replace("[", "").replace("]", "").split(",")
                ]

                j = structure.loc[structure.residue_no == pairing[0]].iloc[0]
                edge_indices.append(
                    (
                        int(i.residue_no),
                        int(pairing[0]),
                    )
                )
                if 'distance' in EXTENDED_PAIRINGS_FEATURES:
                    edge_attr.append(
                        (
                            pairing_type,
                            abs(
                                ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                                ** 0.5
                            ),
                            (
                                abs(i.residue_no - j.residue_no)
                                if i["residue"].split(".")[0] == j["residue"].split(".")[0]
                                else 0
                            ), 
                        )
                        + tuple(pairing[1:])
                    )
                else:   
                    edge_attr.append(
                        (
                            pairing_type,
                            abs(
                                ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                                ** 0.5
                            )
                        )
                        + tuple(pairing[1:])
                    )
                checked.append(pairing[0])
                continue
        for _, j in structure.iterrows():
            if j.residue_no in checked:
                continue
            if i.residue_no != j.residue_no and (
                abs(i.residue_no - j.residue_no) == 1
                and i["residue"].split(".")[0] == j["residue"].split(".")[0]
                or abs(((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2) ** 0.5)
                <= float(features_preferences["max_euclidean_distance"])
            ):
                edge_indices.append(
                    (
                        int(i.residue_no),
                        int(j.residue_no),
                    )
                )
                if features_preferences["pairing_features"] == "basic":
                    edge_attr.append(
                        (
                            -1,
                            abs(
                                ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                                ** 0.5
                            ),
                            (
                                abs(i.residue_no - j.residue_no)
                                if i["residue"].split(".")[0]
                                == j["residue"].split(".")[0]
                                else 0
                            ),
                        )
                    )
                if features_preferences["pairing_features"] == "extended":
                    edge_attr.append(
                        (
                            -1,
                            abs(
                                ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                                ** 0.5
                            ),
                            (
                                abs(i.residue_no - j.residue_no)
                                if i["residue"].split(".")[0]
                                == j["residue"].split(".")[0]
                                else 0
                            ),
                        )
                        + tuple(
                            [0.0]
                            * (len(EXTENDED_PAIRINGS_FEATURES) - len(PAIRINGS_FEATURES))
                        )
                    )
    edge_indices = np.array(list(edge_indices))

    if features_preferences["pairing_features"] == "extended":
        edge_attr = np.array(
            [
                np.concatenate(
                    (
                        parse_bp_type_onehot(
                            bp_f[0], copy.deepcopy(default_edge_features), bp_f[1]
                        ),
                        np.array(bp_f[2:]),
                    )
                )
                for bp_f in edge_attr
            ]
        )
    if features_preferences["pairing_features"] == "basic":
        edge_attr = np.array(
            [
                parse_bp_type_onehot(
                    bp_f[0], copy.deepcopy(default_edge_features), bp_f[1]
                )
                for bp_f in edge_attr
            ]
        )
    data = Data(
        x=torch.tensor(
            structure.loc[:, NODE_FEATURE_COLUMNS].values, dtype=torch.float32
        ),
        edge_index=torch.tensor(edge_indices.T, dtype=torch.int64),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(
            RMSD,
            dtype=torch.float32,
        ),
    )

    torch.save(data, os.path.join(output_path, f"{description_name}.pt"))
    return os.path.join(output_path, f"{description_name}.pt")


def prepare_graphs(
    description_name: str,
    features: pd.DataFrame,
    rmsd: float,
    features_preferences: Type[Dict],
    output_path: str = "/tmp/output",
) -> pd.DataFrame:
    os.makedirs(output_path, exist_ok=True)
    prepare_graph_iter(
        description_name, features, features_preferences, rmsd, output_path
    )
    return os.path.join(output_path, f"{description_name}.pt")
