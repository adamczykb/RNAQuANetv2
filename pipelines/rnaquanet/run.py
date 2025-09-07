from pipelines.structure_processing.nodes import (
    extract_features_from_dataset,
)
import argparse
import json

if __name__ == "__main__":
    # Run the pipeline
    argparser = argparse.ArgumentParser(description="RNAQUANet file processing")
    argparser.add_argument(
        "-f",
        "--file_path",
        type=str,
        required=True,
        help="Path to the input file",
    )
    argparser.add_argument(
        "-r",
        "--rmsd",
        type=float,
        default=0.0,
        help="RMSD value for the dataset",
    )
    features_preferences = {
        "tools_path": "/rnaquanet/tools/",
        "basepair_tool": {"name": "x3dna", "path": f"/rnaquanet/tools/x3dna-dssr"},
        "pairing_features": "extended",
        "atom_for_distance_calculations": "C1'",
        "max_euclidean_distance": "16.0",
        "regenerate_features_when_exists": True,
        "na_solution": "ENCODER",
        "BASE_PATH": "/rnaquanet/",
    }
    # features_preferences = {
    #     "tools_path": "/home/adamczykb/rnaquanet/tools",
    #      "basepair_tool": {
    #          "name": "x3dna",
    #          "path": f"/home/adamczykb/rnaquanet/tools/x3dna-dssr",
    #      },
    #     "pairing_features": "extended",
    #     "atom_for_distance_calculations": "C1'",
    #     "max_euclidean_distance": "16.0",
    #     "regenerate_features_when_exists": True,
    #     "na_solution": "ENCODER",
    #     "BASE_PATH": "/home/adamczykb/rnaquanet/pipelines/rnaquanet/",
    # }
    args = argparser.parse_args()

    r = extract_features_from_dataset(
        args.file_path.strip(),
        args.rmsd,
        features_preferences,
    )
    print(json.dumps({"graph_path": r[0], "csv_path": r[1]}, indent=4))
