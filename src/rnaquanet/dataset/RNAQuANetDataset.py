from enum import Enum
import json
import logging
from math import inf
import os
import os.path as osp
import tarfile
from os import listdir
from os.path import isfile, join
from tempfile import NamedTemporaryFile

import requests
import torch
from torch_geometric.data import Dataset, download_url as download_


class RNAQuANetDatasetSubset(Enum):
    ALL = "all"
    SOLO = "solo"
    DNA = "rna-dna hybrid"
    COMPLEX = "complex"


class RNAQuANetDataset(Dataset):
    def __init__(
        self,
        root,
        subsets: list[RNAQuANetDatasetSubset] = None,
        cutoff: float = None,
        download_url: str = None,
        test: bool = False,
        max_rmsd: float = 16.0,
    ):
        self.logging_s = logging.getLogger(__name__)
        self.download_url = download_url
        self.root = root
        self.structures = []
        if test:
            self.structures = [
                f for f in listdir(self.root) if isfile(join(self.root, f))
            ]
        else:
            try:
                for f in listdir(self.root):
                    if (
                        isfile(join(self.root, f))
                        and f.endswith(".pt")
                        and bool(
                            torch.load(osp.join(self.root, f), weights_only=False).y[0]
                            < max_rmsd
                        )
                    ):
                        self.structures.append(f)
            except:
                self.logging_s.warning(
                    "The dataset does not contain the 'y' tensor in the data files. "
                    "Using the 'y' as RMSD."
                )
                for f in listdir(self.root):
                    try:

                        if (
                            isfile(join(self.root, f))
                            and f.endswith(".pt")
                            and bool(
                                torch.load(osp.join(self.root, f), weights_only=False).y
                                < max_rmsd
                            )
                        ):
                            self.structures.append(f)
                    except Exception as e:
                        self.logging_s.error(
                            f"Error while loading the file {f}: {e}. "
                            "Skipping this file."
                        )
                        os.remove(osp.join(self.root, f))
                        continue

        if cutoff or subsets:
            self.label_pdb_property(
                self.subsets if isinstance(subsets, list) else [subsets]
            )
        super().__init__(root)

    @property
    def raw_file_names(self):
        return [".".join(f.split(".")[:-1]) for f in self.structures]

    @property
    def processed_file_names(self):
        return self.structures

    def label_pdb_property(
        self, cutoff: float = 5.0, subsets: list[RNAQuANetDatasetSubset] = None
    ):
        if (
            len(self.structures[0].split("_")[0]) != 4
            and len(self.structures[0].split("_")[0]) != 12
        ):
            raise ValueError(
                "The first part of the structure name should be a PDB ID (4 or 12 characters long)."
            )
        self.pdb_ids = dict()
        for structure in self.structures:
            self.pdb_ids[structure.split("_")[0].lower()] = self.pdb_ids.get(
                structure.split("_")[0].lower(), []
            ) + [structure]

        self.molecule_type = dict()

        r = requests.post(
            "https://data.rcsb.org/graphql",
            json={
                "query": """query Elements($pdb_ids:[String!]!) {
                                entries(entry_ids: $pdb_ids) {
                                    refine{
                                    ls_d_res_high
                                    }
                                    rcsb_entry_info {
                                        polymer_entity_count_protein
                                    }
                                    entry {
                                    id,
                                    
                                    },
                                    struct_keywords {
                                        pdbx_keywords
                                        text
                                    }
                                    exptl {
                                        method, 
                                    }
                                }
                            }
                        """,
                "variables": {"pdb_ids": list(self.pdb_ids.keys())},
            },
        )

        for entry in json.loads(r.text)["data"]["entries"]:
            if (
                "protein" in entry["struct_keywords"]["text"].lower()
                or "ribosom" in entry["struct_keywords"]["text"].lower()
                or "complex" in entry["struct_keywords"]["text"].lower()
                or entry["rcsb_entry_info"]["polymer_entity_count_protein"] > 0
            ):
                self.molecule_type[entry["entry"]["id"].lower()] = [
                    RNAQuANetDatasetSubset.COMPLEX,
                    (entry["refine"][0]["ls_d_res_high"] if entry["refine"] else inf),
                ]

            elif "dna" in entry["struct_keywords"]["text"].lower():
                self.molecule_type[entry["entry"]["id"].lower()] = [
                    RNAQuANetDatasetSubset.DNA,
                    (entry["refine"][0]["ls_d_res_high"] if entry["refine"] else inf),
                ]

            elif "rna" in entry["struct_keywords"]["text"].lower():
                self.molecule_type[entry["entry"]["id"].lower()] = [
                    RNAQuANetDatasetSubset.SOLO,
                    (entry["refine"][0]["ls_d_res_high"] if entry["refine"] else inf),
                ]

            else:
                print(f"unknown: {entry}")
        self.structures = []
        for structure in self.pdb_ids.keys():
            if (
                RNAQuANetDatasetSubset.ALL in subsets
                or self.molecule_type[structure][0] in subsets
            ) and self.molecule_type[structure][1] <= cutoff:
                self.structures += self.pdb_ids[structure]

    def download(self):
        if self.download_url is not None:
            with NamedTemporaryFile(
                suffix=f".{self.download_url.split('.')[-1]}"
            ) as temp:
                download_(
                    self.download_url,
                    "/".join(temp.name.split("/")[:-1]),
                    filename=temp.name.split("/")[-1],
                )
                mode = (
                    f"r:{self.download_url.split('.')[-1]}"
                    if len(self.download_url.split("/")[-1].split(".")) > 2
                    else "rb"
                )
                if "tar" in temp.split("/")[-1].split(".")[-2:]:
                    with tarfile.open(
                        temp.name,
                        mode=mode,
                    ) as tar:
                        tar.extractall(
                            self.root,
                            filter=lambda x: (x if x.name.endswith(".pt") else None),
                        )
                else:
                    raise ValueError("Archive is not tar")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.root, self.processed_file_names[idx]), weights_only=False
        )
        return data


class RNAQuANetDatasetTuplet(RNAQuANetDataset):
    def __init__(
        self,
        root,
        subsets: list[RNAQuANetDatasetSubset] = None,
        cutoff: float = None,
        download_url: str = None,
        test: bool = False,
        size_of_common_identifier: int = 2,
        max_rmsd: float = 16.0,
    ):
        super().__init__(
            root,
            subsets=subsets,
            cutoff=cutoff,
            download_url=download_url,
            test=test,
            max_rmsd=max_rmsd,
        )
        self.tuplets = []
        self.reference_groups = dict()
        self.create_directed_permutations(size_of_common_identifier)

    @property
    def raw_file_names(self):
        return [".".join(f.split(".")[:-1]) for f in self.structures]

    @property
    def processed_file_names(self):
        return self.structures

    def len(self):
        return len(self.tuplets)

    def create_directed_permutations(self, size_of_common_identifier: int = 2):
        if size_of_common_identifier < 1:
            raise ValueError("The size of the common identifier should be at least 1.")
        for i, s in enumerate(self.structures):
            identifier = "_".join(s.split("_")[:size_of_common_identifier])
            if identifier not in self.reference_groups:
                self.reference_groups[identifier] = []
            self.reference_groups[identifier].append(i)
        for k in self.reference_groups.keys():
            if len(self.reference_groups[k]) < 2:
                continue
            for i in range(len(self.reference_groups[k])):
                for j in range(i + 1, len(self.reference_groups[k])):
                    self.tuplets.append(
                        (self.reference_groups[k][i], self.reference_groups[k][j])
                    )

    def get(self, idx):
        if idx >= len(self.tuplets):
            raise IndexError("Index out of range.")
        index1, index2 = self.tuplets[idx]
        data1 = torch.load(
            osp.join(self.root, self.processed_file_names[index1]), weights_only=False
        )
        data2 = torch.load(
            osp.join(self.root, self.processed_file_names[index2]), weights_only=False
        )
        return (data1, data2)
