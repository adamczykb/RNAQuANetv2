from enum import Enum
import json
from math import inf
import os.path as osp
import tarfile
from os import listdir
from os.path import isfile, join
from tempfile import NamedTemporaryFile

import requests
import torch
from torch_geometric.data import Dataset, download_url


class RNAQuANetDatasetSubset(Enum):
    ALL = 'all'
    SOLO = 'solo'
    DNA = 'rna-dna hybrid'
    COMPLEX = 'complex'


class RNAQuANetDataset(Dataset):
    def __init__(
        self,
        root,
        subsets: list[RNAQuANetDatasetSubset]= None,
        cutoff: float = None ,
        download_url: str = None,
        test: bool = False,
    ):
        self.download_url = download_url
        self.subsets = subsets if isinstance(subsets, list) else [subsets]
        self.root = root
        if test:
            self.structures = [f for f in listdir(self.root) if isfile(join(self.root, f))]
        else:
            self.structures = [f for f in listdir(self.root) if isfile(join(self.root, f)) and bool(torch.load(
                osp.join(self.root, f), weights_only=False
            ).y[0]<16)]
        if cutoff or subsets:
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
                    RNAQuANetDatasetSubset.ALL in self.subsets
                    or self.molecule_type[structure][0] in self.subsets
                ) and self.molecule_type[structure][1] <= cutoff:
                    self.structures += self.pdb_ids[structure]
        super().__init__(root)

    @property
    def raw_file_names(self):
        return [".".join(f.split(".")[:-1]) for f in self.structures]

    @property
    def processed_file_names(self):
        return self.structures

    def download(self):
        if self.download_url is not None:
            with NamedTemporaryFile(
                suffix=f".{self.download_url.split('.')[-1]}"
            ) as temp:
                download_url(
                    self.download_url,
                    "/".join(temp.name.split("/")[:-1]),
                    filename=temp.name.split("/")[-1],
                )
                mode = (
                    f"r:{download_url.split('.')[-1]}"
                    if len(download_url.split("/")[-1].split(".")) > 2
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


class RNAQuANetDatasetTriplet(Dataset):
    def __init__(
        self,
        root,
        download_url: str = None,
        subsets: list[RNAQuANetDatasetSubset] = RNAQuANetDatasetSubset.ALL,
        cutoff: float = 2.5,
    ):
        self.download_url = download_url
        self.subsets = subsets if isinstance(subsets, list) else [subsets]
        self.root = root
        self.structures = [f for f in listdir(self.root) if isfile(join(self.root, f))]
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
                RNAQuANetDatasetSubset.ALL in self.subsets
                or self.molecule_type[structure][0] in self.subsets
            ) and self.molecule_type[structure][1] <= cutoff:
                self.structures += self.pdb_ids[structure]
        super().__init__(root)

    @property
    def raw_file_names(self):
        return [".".join(f.split(".")[:-1]) for f in self.structures]

    @property
    def processed_file_names(self):
        return self.structures

    def download(self):
        if self.download_url is not None:
            with NamedTemporaryFile(
                suffix=f".{self.download_url.split('.')[-1]}"
            ) as temp:
                download_url(
                    self.download_url,
                    "/".join(temp.name.split("/")[:-1]),
                    filename=temp.name.split("/")[-1],
                )
                mode = (
                    f"r:{download_url.split('.')[-1]}"
                    if len(download_url.split("/")[-1].split(".")) > 2
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