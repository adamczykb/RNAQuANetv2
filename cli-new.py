import argparse
import csv
import multiprocessing as mp
import requests
import logging
import os


def send_and_receive_pdb_file(
    pdb_file_path__rmsd: list,
    input_dir: str,
    output: str,
):
    logging.info("Starting process...")
    for en, i in enumerate(pdb_file_path__rmsd):
        rmsd_value = i[0]
        description_value = i[1]
        name_value = i[2]
        structure_file_path_pdb = f"{input_dir}/{description_value}"
        if os.path.exists(f"{output}/{name_value}.pt"):
            continue
        with open(structure_file_path_pdb, "rb") as file_obj:
            files = {"file": file_obj}
            data = {"RMSD": rmsd_value}

            response = requests.post(
                "http://127.0.0.1:8080/pdb",
                files=files,
                data=data,
                timeout=10000,
            )
        if response.status_code == 200:
            if not os.path.exists(
                os.path.join(
                    "/".join(
                        os.path.join(f"{output}/{name_value}.pt").split(os.path.sep)[:-1]
                    )
                )
            ):
                os.makedirs(
                    "/".join(
                        os.path.join(f"{output}/{name_value}.pt").split(os.path.sep)[:-1]
                    )  # Create output directory if it does not exist
                )
            with open(f"{output}/{name_value}.pt", "wb") as pt_file_output:
                pt_file_output.write(response.content)
        else:
            logging.error(
                f"Error: {response.status_code} - {response.text} for {description_value}"
            )
        logging.info(
            "Process completed for %s %d/%d",
            description_value,
            en,
            len(pdb_file_path__rmsd),
        )


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="RNAQuANet batch processing")
    parse.add_argument("-i", "--input_dir", type=str, help="Input dir", required=True)
    parse.add_argument("-c", "--csv_input", type=str, help="CSV input", required=True)
    parse.add_argument(
        "-r", "--rmsd_col", type=str, help="RMSD column in csv", required=True
    )
    parse.add_argument(
        "-d", "--description", type=str, help="description column in csv", required=True
    )
    parse.add_argument("-p", "--name", type=str, help="name column in csv")
    parse.add_argument("-o", "--output_dir", type=str, help="Output dir", required=True)
    parse.add_argument(
        "-n", "--process_number", type=str, help="Number of processes", required=True
    )
    args = parse.parse_args()
    entries = []
    done = 0
    with open(args.csv_input, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Access the columns using the names you provided
            rmsd_value = row[args.rmsd_col]
            description_value = row[args.description]
            if args.name:
                name_value = row[args.name]
            else:
                name_value = description_value.split(".")[0]
            # Do something with the values
            if not os.path.exists(os.path.join(args.input_dir, f"{description_value}")):
                print(
                    f"Skipping {os.path.join(args.input_dir, f'{description_value}')} as it does not exist in input directory."
                )
                continue
            if not os.path.exists(f"{args.output_dir}/{name_value}.pt"):
                entries.append([rmsd_value, description_value, name_value])
            else:
                done += 1
                # print(
                #     f"Skipping {description_value} as it already exists in output directory."
                # )
    import random

    random.shuffle(entries)
    entries_splited = []
    split_by = int(args.process_number)
    for i in range(split_by + 1):
        entries_splited.append(
            entries[i * len(entries) // split_by : (i + 1) * len(entries) // split_by]
        )
    print(
        f"Splitting {len(entries)} entries into {len(entries_splited)} processes. Done: {done} entries.",
        flush=True,
    )
    logging.basicConfig(
        filename="rnaquanet.log",
        filemode="a",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )

    with mp.Pool(15) as pool:
        pool.starmap(
            send_and_receive_pdb_file,
            [
                (
                    entries,
                    str(args.input_dir),
                    str(args.output_dir),
                )
                for entries in entries_splited
            ],
        )
    print("All processes completed.")
