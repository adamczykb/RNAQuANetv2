# RNAQuANet

To run the feature extraction web service, please follow the instructions below:
```bash
docker run -p 4545:8080 -d tzok/maxit:server
docker run --publish 8080:8080 adamczykb/rnaquanet_pipeline:full_features
```
To preprocess structures, use the following command:
```bash
python3 ./cli-new.py -i [directory_with_pdb_files] -c [csv_with_name_and_rmsd] -r [RMSD_column] -d [file_path_column] -o [output_directory] -p [structure_name] -n 15
# Example:
python3 ./cli-new.py -i /home/adamczykb/rnaquanet/data/01_raw/lociparse_structures/val -c /home/adamczykb/rnaquanet/data/01_raw/lociparse_structures/val.csv -r "RMSD" -d "description" -o /home/adamczykb/rnaquanet/data/06_model_input/lociparse_structures/val -p id  -n 15
```

