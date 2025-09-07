import numpy as np
import pandas as pd
import torch

from rnaquanet.pipelines.structure_processing.descriptive_columns import *
from rnaquanet.settings import BASE_PATH

COMMON_COLUMNS = (
    ["atr_" + i for i in ATR_COMMON_COLUMNS]
    + ["bon_" + i for i in BON_COMMON_COLUMNS]
    + ["aag_" + i for i in AAG_COMMON_COLUMNS]
)
featuresA = (
    ["atr_" + i for i in ATR_A_COLUMNS]
    + ["aag_" + i for i in AAG_A_COLUMNS]
    + ["bon_" + i for i in BON_A_COLUMNS]
)
featuresU = (
    ["atr_" + i for i in ATR_U_COLUMNS]
    + ["aag_" + i for i in AAG_U_COLUMNS]
    + ["bon_" + i for i in BON_U_COLUMNS]
)
featuresG = (
    ["atr_" + i for i in ATR_G_COLUMNS]
    + ["aag_" + i for i in AAG_G_COLUMNS]
    + ["bon_" + i for i in BON_G_COLUMNS]
)
featuresC = (
    ["atr_" + i for i in ATR_C_COLUMNS]
    + ["aag_" + i for i in AAG_C_COLUMNS]
    + ["bon_" + i for i in BON_C_COLUMNS]
)


def compress_adeine_features(df: pd.DataFrame) -> pd.DataFrame:
    if torch.cuda.is_available():
        adeine_compressor = torch.jit.load(f"{BASE_PATH}/models/modelAAutoEncoder.pt")
    else:
        adeine_compressor = torch.jit.load(
            f"{BASE_PATH}/models/modelAAutoEncoderCPU.pt"
        )
    results_common = []
    results_features = []
    for description_name, group in df.groupby("description"):
        group = group.sort_values("residue_no")
        a = group.loc[:, featuresA + COMMON_COLUMNS + ["nucleotide"]]
        a.loc[:, featuresA + COMMON_COLUMNS] = (
            a.loc[:, featuresA + COMMON_COLUMNS].replace("-", np.nan).astype(float)
        )
        pd.set_option("future.no_silent_downcasting", True)
        a = a.ffill()
        a = a.bfill()
        group_filled = group.ffill()
        group_filled = group_filled.bfill()
        group.loc[:, ~group.columns.isin(PAIRINGS_FEATURES)] = group_filled.loc[
            :, ~group_filled.columns.isin(PAIRINGS_FEATURES)
        ]
        results_common.append(
            pd.concat(
                [
                    a.loc[a.nucleotide == "A", COMMON_COLUMNS],
                    group.loc[
                        group.nucleotide == "A",
                        ~group.columns.isin(featuresA + COMMON_COLUMNS),
                    ],
                ],
                axis=1,
            )
        )
        results_features.extend(a.loc[a.nucleotide == "A", featuresA].values)
    other_col = list(
        (
            set(df.columns)
            - set(featuresA)
            - set(featuresC)
            - set(featuresG)
            - set(featuresU)
        )
        - set(COMMON_COLUMNS)
    )
    df = pd.DataFrame(
        pd.concat(results_common, axis=0),
        columns=COMMON_COLUMNS + other_col,
    )
    compressed_features = []
    results_features = np.array(results_features)
    for batch in np.array_split(
        results_features,
        indices_or_sections=results_features.shape[0]
        // min(1024, results_features.shape[0]),
        axis=0,
    ):
        if torch.cuda.is_available():
            compressed_features.append(
                np.asarray(
                    adeine_compressor(torch.Tensor(batch).to("cuda"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
        else:
            compressed_features.append(
                np.asarray(
                    adeine_compressor(torch.Tensor(batch).to("cpu"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
        
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                np.concatenate(compressed_features, axis=0),
                columns=["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7"],
            ).set_index(df.index),
        ],
        axis=1,
    )
    return df


def compress_cytosine_features(df: pd.DataFrame) -> pd.DataFrame:
    if torch.cuda.is_available():
        cytosine_compressor = torch.jit.load(f"{BASE_PATH}/models/modelCAutoEncoder.pt")
    else:
        cytosine_compressor = torch.jit.load(
            f"{BASE_PATH}/models/modelCAutoEncoderCPU.pt"
        )

    results_common = []
    results_features = []
    for description_name, group in df.groupby("description"):
        group = group.sort_values("residue_no")
        a = group.loc[:, featuresC + COMMON_COLUMNS + ["nucleotide"]]
        a.loc[:, featuresC + COMMON_COLUMNS] = (
            a.loc[:, featuresC + COMMON_COLUMNS].replace("-", np.nan).astype(float)
        )
        pd.set_option("future.no_silent_downcasting", True)
        a = a.ffill()
        a = a.bfill()
        group_filled = group.ffill()
        group_filled = group_filled.bfill()
        group.loc[:, ~group.columns.isin(PAIRINGS_FEATURES)] = group_filled.loc[
            :, ~group_filled.columns.isin(PAIRINGS_FEATURES)
        ]
        results_common.append(
            pd.concat(
                [
                    a.loc[a.nucleotide == "C", COMMON_COLUMNS],
                    group.loc[
                        group.nucleotide == "C",
                        ~group.columns.isin(featuresC + COMMON_COLUMNS),
                    ],
                ],
                axis=1,
            )
        )
        results_features.extend(a.loc[a.nucleotide == "C", featuresC].values)
    other_col = list(
        (
            set(df.columns)
            - set(featuresA)
            - set(featuresC)
            - set(featuresG)
            - set(featuresU)
        )
        - set(COMMON_COLUMNS)
    )
    df_r = pd.DataFrame(
        pd.concat(results_common, axis=0),
        columns=COMMON_COLUMNS + other_col,
    )
    compressed_features = []
    results_features = np.array(results_features)

    for batch in np.array_split(
        results_features,
        indices_or_sections=results_features.shape[0]
        // min(1024, results_features.shape[0]),
        axis=0,
    ):
        if torch.cuda.is_available():
            compressed_features.append(
                np.asarray(
                    cytosine_compressor(torch.Tensor(batch).to("cuda"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
        else:
            compressed_features.append(
                np.asarray(
                    cytosine_compressor(torch.Tensor(batch).to("cpu"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
    df_r = pd.concat(
        [
            df_r,
            pd.DataFrame(
                np.concatenate(compressed_features, axis=0),
                columns=["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7"],
            ).set_index(df_r.index),
        ],
        axis=1,
    )
    return df_r


def compress_uracil_features(df: pd.DataFrame) -> pd.DataFrame:
    if torch.cuda.is_available():
        uracil_compressor = torch.jit.load(f"{BASE_PATH}/models/modelUAutoEncoder.pt")
    else:
        uracil_compressor = torch.jit.load(
            f"{BASE_PATH}/models/modelUAutoEncoderCPU.pt"
        )
    results_common = []
    results_features = []
    for description_name, group in df.groupby("description"):
        group = group.sort_values("residue_no")
        a = group.loc[:, featuresU + COMMON_COLUMNS + ["nucleotide"]]
        a.loc[:, featuresU + COMMON_COLUMNS] = (
            a.loc[:, featuresU + COMMON_COLUMNS].replace("-", np.nan).astype(float)
        )
        pd.set_option("future.no_silent_downcasting", True)
        a = a.ffill()
        a = a.bfill()
        group_filled = group.ffill()
        group_filled = group_filled.bfill()
        group.loc[:, ~group.columns.isin(PAIRINGS_FEATURES)] = group_filled.loc[
            :, ~group_filled.columns.isin(PAIRINGS_FEATURES)
        ]
        results_common.append(
            pd.concat(
                [
                    a.loc[a.nucleotide == "U", COMMON_COLUMNS],
                    group.loc[
                        group.nucleotide == "U",
                        ~group.columns.isin(featuresU + COMMON_COLUMNS),
                    ],
                ],
                axis=1,
            )
        )
        results_features.extend(a.loc[a.nucleotide == "U", featuresU].values)
    other_col = list(
        (
            set(df.columns)
            - set(featuresA)
            - set(featuresC)
            - set(featuresG)
            - set(featuresU)
        )
        - set(COMMON_COLUMNS)
    )
    df = pd.DataFrame(
        pd.concat(results_common, axis=0),
        columns=COMMON_COLUMNS + other_col,
    )
    compressed_features = []
    results_features = np.array(results_features)
    for batch in np.array_split(
        results_features,
        indices_or_sections=results_features.shape[0]
        // min(1024, results_features.shape[0]),
        axis=0,
    ):

        if torch.cuda.is_available():
            compressed_features.append(
                np.asarray(
                    uracil_compressor(torch.Tensor(batch).to("cuda"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
        else:
            compressed_features.append(
                np.asarray(
                    uracil_compressor(torch.Tensor(batch).to("cpu"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                np.concatenate(compressed_features, axis=0),
                columns=["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7"],
            ).set_index(df.index),
        ],
        axis=1,
    )
    return df


def compress_guanine_features(df: pd.DataFrame) -> pd.DataFrame:
    if torch.cuda.is_available():
        guanine_compressor = torch.jit.load(f"{BASE_PATH}/models/modelGAutoEncoder.pt")
    else:
        guanine_compressor = torch.jit.load(
            f"{BASE_PATH}/models/modelGAutoEncoderCPU.pt"
        )
    results_common = []
    results_features = []
    for description_name, group in df.groupby("description"):
        group = group.sort_values("residue_no")
        a = group.loc[:, featuresG + COMMON_COLUMNS + ["nucleotide"]]
        a.loc[:, featuresG + COMMON_COLUMNS] = (
            a.loc[:, featuresG + COMMON_COLUMNS].replace("-", np.nan).astype(float)
        )
        pd.set_option("future.no_silent_downcasting", True)
        a = a.ffill()
        a = a.bfill()
        group_filled = group.ffill()
        group_filled = group_filled.bfill()
        group.loc[:, ~group.columns.isin(PAIRINGS_FEATURES)] = group_filled.loc[
            :, ~group_filled.columns.isin(PAIRINGS_FEATURES)
        ]
        results_common.append(
            pd.concat(
                [
                    a.loc[a.nucleotide == "G", COMMON_COLUMNS],
                    group.loc[
                        group.nucleotide == "G",
                        ~group.columns.isin(featuresG + COMMON_COLUMNS),
                    ],
                ],
                axis=1,
            )
        )
        results_features.extend(a.loc[a.nucleotide == "G", featuresG].values)
    other_col = list(
        (
            set(df.columns)
            - set(featuresA)
            - set(featuresC)
            - set(featuresG)
            - set(featuresU)
        )
        - set(COMMON_COLUMNS)
    )
    df = pd.DataFrame(
        pd.concat(results_common, axis=0),
        columns=COMMON_COLUMNS + other_col,
    )
    compressed_features = []
    results_features = np.array(results_features)
    for batch in np.array_split(
        results_features,
        indices_or_sections=results_features.shape[0]
        // min(1024, results_features.shape[0]),
        axis=0,
    ):
        if torch.cuda.is_available():
            compressed_features.append(
                np.asarray(
                    guanine_compressor(torch.Tensor(batch).to("cuda"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
        else:
            compressed_features.append(
                np.asarray(
                    guanine_compressor(torch.Tensor(batch).to("cpu"))
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                np.concatenate(compressed_features, axis=0),
                columns=["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7"],
            ).set_index(df.index),
        ],
        axis=1,
    )
    return df
