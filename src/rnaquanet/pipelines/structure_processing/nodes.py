import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import tarfile
import copy
from contextlib import closing
from typing import Dict, List, Type
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO
import numpy as np
import pandas as pd
import requests
import torch
from kedro.io.core import get_protocol_and_path
from torch_geometric.data import Data
from tqdm import tqdm

from rnaquanet.pipelines.structure_processing.descriptive_columns import (
    EDGE_FEATURE_COLUMNS,
    EXTENDED_PAIRINGS_FEATURES,
    NODE_FEATURE_COLUMNS,
    PAIRINGS_FEATURES,
)
from rnaquanet.pipelines.structure_processing.feature_process import (
    clear_na_features,
    extract_nucleotides,
    extract_pairing,
    process_features,
)
from rnaquanet.pipelines.structure_processing.parallel_feature_extraction import (
    extract_features,
)
from rnaquanet.settings import BASE_PATH


def prepare_raw_data(filepath: str, dataset_name_param: str) -> str:
    """Download, extract and clean old data for further processing.

    Args:
        filepath: tar.gz dataset url
        name: dataset name
    Returns:
        tuple: (train_exists, val_exists, test_exists)
    """
    try:
        if not os.path.exists(f"data/01_raw/{dataset_name_param}"):
            os.makedirs(f"data/01_raw/{dataset_name_param}")
            protocol, path = get_protocol_and_path(filepath)

            archive_download_destination = f"data/01_raw/{dataset_name_param}/{dataset_name_param}.{'.'.join(path.split('/')[-1].split('.')[1:])}"

            if protocol in ["http", "https"]:
                response = requests.get(f"{protocol}://{path}", stream=True, timeout=10)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
                    with open(
                        archive_download_destination,
                        "wb",
                    ) as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
            elif protocol == "" and path != "":
                shutil.copy(
                    path,
                    archive_download_destination,
                )
            elif path == "":
                return f"data/01_raw/{dataset_name_param}/"
            else:
                raise ValueError("Url protocol is unknown")
            mode = (
                f"r:{path.split('.')[-1]}"
                if len(path.split("/")[-1].split(".")) > 2
                else "rb"
            )
            if "tar" in path.split("/")[-1].split(".")[-2:]:
                with tarfile.open(
                    archive_download_destination,
                    mode=mode,
                ) as tar:
                    tar.extractall(
                        f"data/01_raw/{dataset_name_param}",
                        filter=lambda tar_info, path: (
                            tar_info
                            if tar_info.name.endswith(".pdb")
                            or tar_info.name.endswith(".csv")
                            else None
                        ),
                    )
            else:
                raise ValueError("Archive is not tar")

    except tarfile.ReadError:
        shutil.rmtree(f"data/01_raw/{dataset_name_param}")
        logging.error("Tar read error")
        raise
    return f"data/01_raw/{dataset_name_param}/"


def extract_pairing_nucleobase(
    preprocessing_source_path,
    entry,
    features_path: str,
    tool_url_name: str,
    atom_for_distance_calculations: str,
) -> np.array:

    if os.path.exists(os.path.join(preprocessing_source_path, entry)):
        if os.path.join(preprocessing_source_path, entry).split(".")[-1] == "cif":
            parser = MMCIFParser(QUIET=True)
            io = MMCIFIO()
        elif os.path.join(preprocessing_source_path, entry).split(".")[-1] == "pdb":
            parser = PDBParser(QUIET=True)
            io = PDBIO()
        else:
            raise ValueError(
                "File type is not supported. Only .pdb and .cif files are supported."
            )
        # task = subprocess.Popen(
        #     f"pdbfixer {os.path.join(BASE_PATH,'../../' ,os.path.join(preprocessing_source_path, entry))} --output={os.path.join(BASE_PATH,'../../' ,os.path.join(preprocessing_source_path, entry+'tmp'))} --replace-nonstandard --add-atoms=all".split(),
        #     stderr=subprocess.PIPE,
        #     stdout=subprocess.PIPE,
        # )
        # r, e = task.communicate()
        # # if task.returncode != 0:
        # #     raise ValueError(
        # #         f"pdbfixer failed {os.path.join(BASE_PATH ,os.path.join(preprocessing_source_path, entry))}"
        # #         + e.decode("utf-8")
        # #     )
        # if (
        #     task.returncode != 0
        #     and os.path.getsize(
        #         os.path.join(
        #             BASE_PATH,
        #             "../../",
        #             os.path.join(preprocessing_source_path, entry + "tmp"),
        #         )
        #     )
        #     > 10
        # ):
        #     os.remove(
        #         os.path.join(
        #             BASE_PATH,
        #             "../../",
        #             os.path.join(preprocessing_source_path, entry),
        #         )
        #     )
        #     os.rename(
        #         os.path.join(
        #             BASE_PATH,
        #             "../../",
        #             os.path.join(preprocessing_source_path, entry + "tmp"),
        #         ),
        #         os.path.join(
        #             BASE_PATH,
        #             "../../",
        #             os.path.join(preprocessing_source_path, entry),
        #         ),
        #     )
        # else:
        #     os.remove(
        #         os.path.join(
        #             BASE_PATH,
        #             "../../",
        #             os.path.join(preprocessing_source_path, entry + "tmp"),
        #         )
        #     )
        used_chains = []
        try:
            structure = parser.get_structure(
                "str", os.path.join(preprocessing_source_path, entry)
            )[0]
        except:
            logging.error(
                f"Error parsing structure {os.path.join(preprocessing_source_path, entry)}"
            )
            os.remove(os.path.join(preprocessing_source_path, entry))
            shutil.copy(
                f'/home/adamczykb/rnaquanet/data/00_reference/lociparse/train/{entry.split("/")[0]}/model.pdb',
                os.path.join(preprocessing_source_path, entry),
            )
            structure = parser.get_structure(
                "str", os.path.join(preprocessing_source_path, entry)
            )[0]
            # raise ValueError(
            #     f"Error parsing structure {os.path.join(preprocessing_source_path, entry)}"
            # )
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
        io.save(os.path.join(preprocessing_source_path, entry))
        temp_df = pd.concat(
            [
                extract_nucleotides(
                    os.path.join(preprocessing_source_path, entry),
                    atom_for_distance_calculations,
                ),
                extract_pairing(
                    os.path.join(preprocessing_source_path, entry), tool_url_name
                ),
            ],
            axis=1,
        )
        temp_df["description"] = os.path.splitext(entry)[0]
        temp_df["residue_no"] = np.linspace(
            0, len(temp_df) - 1, len(temp_df), dtype=int
        )
        return [
            [
                os.path.join(preprocessing_source_path, entry),
                os.path.join(features_path, os.path.splitext(entry)[0]),
            ],
            temp_df,
        ]

    else:
        raise ValueError(
            f"File {os.path.join(preprocessing_source_path, entry)} does not exist"
        )


def extract_features_from_dataset(
    dataset_name_param: str,
    unarchived_dataset_path: str,
    source_directory: str,
    source_entities: pd.DataFrame,
    file_name_column: str,
    features_preferences: dict,
):
    """
    Extracts features from specified dataset.

    Args:
    - dataset_name_param - dataset directory name
    - filtered_file_path - PDB file absolute path

    Returns:
    - features_file_path

    Exceptions:
    - if 'preprocessing/filtered' directory does not exist
    - if any of the files in 'pdb_filepaths' does not exist
    - if tools/RNAgrowth does not exist
    """

    preprocessing_source_path = os.path.join(
        unarchived_dataset_path, source_directory
    )  # directory of pdb files

    features_path = os.path.join(
        "data", "02_intermediate", dataset_name_param, source_directory
    )
    processed_features_path = os.path.join(
        "data", "03_primary", dataset_name_param, f"{source_directory}_processed.csv"
    )
    if not features_preferences["regenerate_features_when_exists"] and os.path.exists(
        processed_features_path
    ):
        df = pd.read_csv(processed_features_path, sep=",")
    else:
        os.makedirs(features_path, exist_ok=True)
        results = []
        with closing(mp.Pool(10)) as pool:
            results = pool.starmap(
                extract_pairing_nucleobase,
                tqdm(
                    zip(
                        np.repeat(preprocessing_source_path, len(source_entities)),
                        np.array(source_entities[:][file_name_column]),
                        np.repeat(features_path, len(source_entities)),
                        np.repeat(
                            features_preferences["basepair_tool"],
                            len(source_entities),
                        ),
                        np.repeat(
                            features_preferences["atom_for_distance_calculations"],
                            len(source_entities),
                        ),
                    ),
                    unit="f",
                    total=len(source_entities),
                ),
            )
        df = pd.concat(
            [i[1] for i in results],
            axis=0,
        )
        df.set_index(["description", "residue_no"], inplace=True, drop=False)
        extract_features([i[0] for i in results], features_preferences)
    return df


def process_features_from_dataset_f(entry, features_path):
    features_file_path = os.path.join(features_path, os.path.splitext(entry)[0])
    temp_df = process_features(features_file_path)
    temp_df["description"] = os.path.splitext(entry)[0]
    temp_df["residue_no"] = range(0, len(temp_df))
    temp_df.set_index(["description", "residue_no"], inplace=True, drop=False)
    return temp_df


def process_features_from_dataset(
    dataset_name_param,
    dataset_subdirectory,
    source_entities,
    features_preferences,
    file_name_column,
    structure_features,
) -> pd.DataFrame:
    """
    Processes features from a dataset and returns a DataFrame containing the combined features.

    Args:
        dataset_name_param (str): The name of the dataset.
        dataset_subdirectory (str): The type [train,test,val] within the dataset.
        source_entities (pd.DataFrame): DataFrame containing source data entities information.
        features_preferences (dict): Dictionary containing preferences for feature processing from yml file.
        file_name_column (str): The column name in source_entities that contains structure names.
        structure_features (pd.DataFrame): DataFrame containing structure features.

    Returns:
        pd.DataFrame: A DataFrame containing the combined features from the dataset and structure features.
    """

    structure_features.set_index(["description", "residue_no"], inplace=True, drop=True)
    features_path = os.path.join(
        "data", "02_intermediate", dataset_name_param, dataset_subdirectory
    )
    full_features_path = os.path.join(
        "data",
        "04_feature",
        dataset_name_param,
        f"{dataset_subdirectory}_full_dataset.csv",
    )
    if not features_preferences["regenerate_features_when_exists"] and os.path.exists(
        full_features_path
    ):
        return pd.read_csv(full_features_path, sep=",")
    with closing(mp.Pool(mp.cpu_count())) as pool:
        results = pool.starmap(
            process_features_from_dataset_f,
            tqdm(
                zip(
                    np.array(source_entities[:][file_name_column]),
                    np.repeat(features_path, len(source_entities)),
                ),
                unit="f",
                total=len(source_entities),
            ),
        )

    df = pd.concat(
        results,
        axis=0,
    )
    df = pd.concat([structure_features, df], axis=1)
    return df


def compress_features(
    features: pd.DataFrame, features_preferences: Type[Dict], source:str, dataset_name_param: str
) -> pd.DataFrame:
    if os.path.exists(
        os.path.join(
            "data",
            "05_feature_compressed",
            dataset_name_param,
            f"{source}_compressed.csv",
        )
    ):
        return pd.read_csv(
            os.path.join(
                "data",
                "05_feature_compressed",
                dataset_name_param,
                f"{source}_compressed.csv",
            ),
        )
    return clear_na_features(features, features_preferences)


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
    entry_feature_vector[EDGE_FEATURE_COLUMNS.index("distance")] = distance

    return entry_feature_vector


def prepare_graph_iter(
    description_name,
    structure,
    features_preferences: Type[Dict],
    source_entities_RMSD: float,
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
            if pd.isna(i[pairing_type]):
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
                        abs(i.residue_no - j.residue_no) if i['residue'].split('.')[0] == j['residue'].split('.')[0] else 0,
                    )
                )
                checked.append(i[pairing_type])
                continue
            if features_preferences["pairing_features"] == "extended":
                pairing = [
                    float(el)
                    for el in i[pairing_type]
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                ]

                j = structure.loc[structure.residue_no == pairing[0]].iloc[0]
                edge_indices.append(
                    (
                        int(i.residue_no),
                        int(pairing[0]),
                    )
                )
                edge_attr.append(
                    (
                        pairing_type,
                        abs(
                            ((i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2)
                            ** 0.5
                        ),
                        abs(i.residue_no - j.residue_no) if i['residue'].split('.')[0] == j['residue'].split('.')[0] else 0
                    )
                    + tuple(pairing[1:])
                )
                checked.append(pairing[0])
                continue
        for _, j in structure.iterrows():
            if j.residue_no in checked:
                continue
            if i.residue_no != j.residue_no and (
                abs(i.residue_no - j.residue_no) == 1 and i['residue'].split('.')[0] == j['residue'].split('.')[0]
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
                            abs(i.residue_no - j.residue_no) if i['residue'].split('.')[0] == j['residue'].split('.')[0] else 0
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
                            abs(i.residue_no - j.residue_no) if i['residue'].split('.')[0] == j['residue'].split('.')[0] else 0
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
            source_entities_RMSD,
            dtype=torch.float32,
        ),
    )

    torch.save(data, os.path.join(output_path, f"{description_name}.pt"))
    return os.path.join(output_path, f"{description_name}.pt")


def prepare_graphs(
    features: pd.DataFrame,
    dataset_name_param: str,
    dataset_subdirectory: str,
    features_preferences: Type[Dict],
    source_entities: pd.DataFrame,
    file_name_column: str,
    rmsd_value_column: str,
) -> pd.DataFrame:

    output_path = os.path.join(
        "data", "06_model_input", dataset_name_param, dataset_subdirectory
    )
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    source_entities[file_name_column] = source_entities[file_name_column].apply(
        lambda x: "".join(x.split(".")[:-1])
        # lambda x: str(os.path.splitext(os.path.basename(x))[0])
    )
    for description_name, structure in tqdm(features.groupby("description")):
        prepare_graph_iter(
            description_name,
            structure,
            features_preferences,
            source_entities.loc[
                source_entities[file_name_column] == description_name,
                [rmsd_value_column],
            ].values[0],
            output_path,
        )

    # with closing(mp.Pool(mp.cpu_count())) as pool:
    #     results = pool.starmap(
    #         prepare_graph_iter,
    #         tqdm(
    #             [
    #                 tuple(
    #                     [
    #                         description_name,
    #                         structure,
    #                         features_preferences,
    #                         source_entities.loc[
    #                             source_entities[file_name_column] == description_name,
    #                             [rmsd_value_column],
    #                         ].values[0],
    #                         output_path,
    #                     ]
    #                 )
    #                 for description_name, structure in features.groupby("description")
    #             ],
    #             unit="f",
    #             total=len(source_entities),
    #         ),
    #     )

    return len(source_entities)
