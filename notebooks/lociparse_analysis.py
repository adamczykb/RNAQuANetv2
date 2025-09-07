import marimo

__generated_with = "0.12.4"
app = marimo.App(width="full")


@app.cell
def _():
    from Bio.PDB import PDBParser
    from minineedle import needle, smith, core
    from Bio.PDB import MMCIFParser
    from Bio.PDB import MMCIFIO
    from Bio.PDB import PDBIO
    from Bio import PDB
    from pymol import cmd
    import os
    import pandas as pd
    from tqdm  import tqdm
    import tempfile
    import requests
    import json
    import matplotlib.pyplot as plt
    import subprocess
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    import marimo as mo
    directory_home = "/home/adamczykb/"
    return (
        MMCIFIO,
        MMCIFParser,
        PDB,
        PDBFile,
        PDBFixer,
        PDBIO,
        PDBParser,
        cmd,
        core,
        directory_home,
        json,
        mo,
        needle,
        os,
        pd,
        plt,
        requests,
        smith,
        subprocess,
        tempfile,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""### Generate sequences from PDB files""")
    return


@app.cell
def _(cmd, os, tqdm):
    for directory in tqdm(
        os.listdir("/home/adamczykb/rnaquanet/data/00_reference/lociparse_train")
    ):
        cmd.reinitialize()
        if os.path.isdir(
            f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}"
        ) and not os.path.exists(
            f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}/model.fasta"
        ):
            cmd.load(
                f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}/model.pdb"
            )
            cmd.save(
                f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}/model.fasta"
            )
    return (directory,)


@app.cell
def _(mo):
    mo.md(r"""### Read sequences from FASTA files and put into pandas""")
    return


@app.cell
def _(os, tqdm):
    models=[]
    for directory in tqdm(
        os.listdir("/home/adamczykb/rnaquanet/data/00_reference/lociparse_train")
    ):
        if os.path.isdir(
            f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}"
        ) and os.path.exists(
            f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}/model.fasta"
        ):
            with open(
                f"/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/{directory}/model.fasta",
                "r",
            ) as f:
                lines = f.readlines()
                sequence=""
                once=True
                for line in lines:
                    if line.startswith(">"):
                        if once:
                            once=False
                            continue
                        raise ValueError(directory+' second chain')

                    sequence+=line.strip()
                models.append(
                    {
                        "id": directory,
                        "sequence": sequence,
                        "length": len(sequence),
                    }
                )
    return directory, f, line, lines, models, once, sequence


@app.cell
def _(mo):
    mo.md("""### Load conversion dictionary for modified residues""")
    return


@app.cell
def _(json, requests, tempfile):
    non_standard=[ 'A23', 'A2L', 'A2M', 'A39', 'A3P', 'A44', 'A5O', 'A6A', 'A7E', 'A9Z', 'ADI', 'ADP', 'AET', 'AMD', 'AMO', 'AP7', 'AVC', 'MA6', 'MAD', 'MGQ', 'MIA', 'MTU', 'M7A', '26A', '2MA', '6IA', '6MA', '6MC', '6MP', '6MT', '6MZ', '6NW', 'F3N', 'N79', 'RIA', 'V3L', 'ZAD', '31H', '31M', '7AT', 'O2Z', 'SRA', '00A', '45A', '8AN', 'LCA', 'P5P', 'PPU', 'PR5', 'PU', 'T6A', 'TBN', 'TXD', 'TXP', '12A', '1MA', '5FA', 'A6G', 'E6G', 'E7G', 'EQ4', 'IG', 'IMP', 'M2G', 'MGT', 'MGV', 'MHG', 'QUO', 'YG', 'YYG', '23G', '2EG', '2MG', '2SG', 'B8K', 'B8W', 'B9B', 'BGH', 'N6G', 'RFJ', 'ZGU', '7MG', 'CG1', 'G1G', 'G25', 'G2L', 'G46', 'G48', 'G7M', 'GAO', 'GDO', 'GDP', 'GH3', 'GNG', 'GOM', 'GRB', 'GTP', 'KAG', 'KAK', 'O2G', 'OMG', '8AA', '8OS', 'LG', 'PGP', 'P7G', 'TPG', 'TG', 'XTS', '102', '18M', '1MG', 'A5M', 'A6C', 'E3C', 'IC', 'M4C', 'M5M', '6OO', 'B8Q', 'B8T', 'B9H', 'JMH', 'N5M', 'RPC', 'RSP', 'RSQ', 'ZBC', 'ZCY', '73W', 'C25', 'C2L', 'C31', 'C43', 'C5L', 'CBV', 'CCC', 'CH', 'CSF', 'OMC', 'S4C', '4OC', 'LC', 'LHH', 'LV2', 'PMT', 'TC', '10C', '1SC', '5HM', '5IC', '5MC', 'A6U', 'IU', 'I4U', 'MEP', 'MNU', 'U25', 'U2L', 'U2P', 'U31', 'U34', 'U36', 'U37', 'U8U', 'UAR', 'UBB', 'UBD', 'UD5', 'UPV', 'UR3', 'URD', 'US5', 'UZR', 'UMO', 'U23', '2AU', '2MU', '2OM', 'B8H', 'FHU', 'FNU', 'F2T', 'RUS', 'ZBU', '3AU', '3ME', '3MU', '3TD', '70U', '75B', 'CNU', 'OMU', 'ONE', 'S4U', 'SSU', 'SUR', '4SU', '85Y', 'DHU', 'H2U', 'LHU', 'PSU', 'PYO', 'P4U', 'T31', '125', '126', '127', '1RN', '5BU', '5FU', '5MU', '9QV', '5GP',"GTA","F86","I","ATP","DOC","GMP","GP3","CSL","M7G","C5P","RY","LCC","AMP","G5J","CFL","UFT"]
    parser=dict()
    res=requests.get('https://rna.bgsu.edu/modified/modified_to_change_data.json')
    p_file=''
    with tempfile.NamedTemporaryFile(suffix='.json',delete=False) as par:
        par.write(res.content)
        p_file=par.name
    with open(p_file,'r') as par_f:
        parser = json.load(par_f)
    parser['DU']={'standard_base':['U']}
    parser['N']={'standard_base':['*']}
    parser['ADP']={'standard_base':['A']}
    parser['UFT']={'standard_base':['U']}
    return non_standard, p_file, par, par_f, parser, res


@app.cell
def _():
    ### Find chain of model for given PDB ID
    return


@app.cell
def _(
    MMCIFParser,
    models_df,
    needle,
    non_standard,
    parser,
    requests,
    tempfile,
    tqdm,
):
    def parse_modified(chain):
        result = []
        rejected = []
        for r in chain:
            if r.get_resname() in ["A", "C", "G", "U", "N", "DU"] + non_standard:
                if r.get_resname() in ["A", "C", "G", "U"]:
                    result.append(r.get_resname())
                else:
                    result.append(parser[r.get_resname()]["standard_base"][0])
            else:
                rejected.append(r.get_resname())
        return "".join(result)


    for l in tqdm(
        models_df.loc[models_df["ture_chain"] == ""].groupby("structure")
    ):
        with tempfile.NamedTemporaryFile(suffix=".cif") as cif:
            response = requests.get(f"https://files.rcsb.org/download/{l[0]}.cif")
            if response.status_code != 200:
                raise Exception(response)
            cif.write(response.content)
            cif.seek(0)
            structure = MMCIFParser(QUIET=True).get_structure("str", cif.name)[0]
            for c in l[1].groupby("common_structure"):
                seq = []
                found = False
                for chain in structure:
                    alignment = needle.NeedlemanWunsch(
                        list(set(c[1]["sequence"]))[0], parse_modified(chain)
                    )
                    alignment.align()
                    if list(set(c[1]["sequence"]))[0] in parse_modified(chain):
                        models_df.loc[
                            (models_df["common_structure"] == c[0]), "ture_chain"
                        ] = chain.id
                        found = True
                        break
                    elif (
                        abs(
                            len(list(set(c[1]["sequence"]))[0])
                            - alignment.get_score()
                        )
                        < 2
                    ):
                        models_df.loc[
                            (models_df["common_structure"] == c[0]), "ture_chain"
                        ] = chain.id
                        found = True
                        break
                    else:
                        seq.append(parse_modified(chain))
                best = seq
                if not found:
                    print(
                        l[0]
                        + " "
                        + c[0]
                        + " "
                        + ",".join(list(set(c[1]["sequence"])))
                        + " "
                        + ",".join(seq)
                    )
    return (
        alignment,
        best,
        c,
        chain,
        cif,
        found,
        l,
        parse_modified,
        response,
        seq,
        structure,
    )


@app.cell
def _(PDB, non_standard):
    class ChainsSelect(PDB.Select):
        chains = ""

        def __init__(self, c):
            self.chains = c

        def accept_residue(self, residue):
            return (
                residue.get_parent().id in self.chains
                and residue.get_resname()
                in ["A", "C", "G", "U", "N", "DU"] + non_standard
            )

    class ResidueSelect(PDB.Select):
        def __init__(self, r):
            self.residues = [x.get_id()[1] for x in r]

        def accept_residue(self, residue):
            return residue.get_id()[1] in self.residues


    class REMOVEOP11OP2FirstSelect(PDB.Select):
        def __init__(self):
            pass

        def accept_atom(self, atom):
            if atom.get_parent().get_id()[1] == 1:
                return atom.get_name() not in ["OP1", "OP2"]
            else:
                return True
    return ChainsSelect, REMOVEOP11OP2FirstSelect, ResidueSelect


@app.cell
def _(mo):
    mo.md(r"""### Declare aligner for finding longest common subsequence""")
    return


@app.cell
def _():
    from Bio import Align
    from Bio.Seq import Seq

    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = -12
    aligner.end_gap_score = 0
    aligner.mode = "global"
    return Align, Seq, aligner


@app.cell
def _(
    ChainsSelect,
    Excpetion,
    MMCIFIO,
    MMCIFParser,
    PDBFile,
    PDBFixer,
    PDBIO,
    PDBParser,
    REMOVEOP11OP2FirstSelect,
    ResidueSelect,
    Seq,
    aligner,
    cmd,
    directory_home,
    non_standard,
    parser,
    pd,
    requests,
    subprocess,
    tempfile,
    tqdm,
):
    def structure_fixer(path):
        # Fixing PDB file, adding necessary atoms and replacing nonstandard residues
        fixer = PDBFixer(filename=path)
        fixer.findNonstandardResidues()
        fixer.missingResidues = {}
        fixer.nonstandardResidues = list(
            [
                (i, parser[i.name]["standard_base"][0])
                for i in fixer.topology.residues()
                if i.name in non_standard
            ]
        )
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        PDBFile.writeFile(fixer.topology, fixer.positions, open(path, "w"))

        # Fixing PDB file that has problem with double OP1 atoms (not solvable in other way)
        if "4K27" in path:
            ref = PDBParser(QUIET=True).get_structure("str", path)[0]
            ref = ref[[i.id for i in ref][0]]
            io = PDBIO()
            io.set_structure(ref)
            io.save(
                path,
                REMOVEOP11OP2FirstSelect(),
            )
        # Fixing PDB file, ordering atoms in cerain RNApuzzle way
        task = subprocess.Popen(
            f"rna_pdb_tools.py --get-rnapuzzle-ready {path} --inplace".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        r, e = task.communicate()
        if task.returncode != 0:
            raise ValueError(f"rna_pdb_tools failed {path}" + e.decode("utf-8"))


    def rmsd_models(inp):
        """
        (source_directory of pdb files, pandas_csv_path_for_storing_results) <- inp
        """
        source, models_df_path = inp
        _models_df = pd.read_csv(models_df_path)
        for ll in tqdm(
            _models_df.loc[_models_df["RMSD"] == 999.0].groupby("common_structure")
        ):
            ref_path = (
                f"{directory_home}rnaquanet/data/00_reference/lociparse_train/{source}/"
                + list(ll[1]["common_structure"])[0]
                + ".pdb"
            )
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdb") as temp_cif:
                    resi = requests.get(
                        f"https://files.rcsb.org/download/{list(set(ll[1]['structure']))[0]}.pdb"
                    )
                    if resi.status_code != 200:
                        raise Excpetion()

                    temp_cif.write(resi.content)
                    temp_cif.seek(0)
                    # structure_fixer(temp_cif.name) # sometimes cleares whole strucutre
                    struct = PDBParser(QUIET=True).get_structure(
                        "str", temp_cif.name
                    )[0]
                    io = PDBIO()
                    io.set_structure(struct)
                    io.save(
                        temp_cif.name,
                        ChainsSelect(list(set(ll[1]["ture_chain"]))),
                    )
                    cmd.reinitialize()
                    cmd.load(
                        temp_cif.name,
                        object="model_",
                    )
                    cmd.save(ref_path)
                structure_fixer(ref_path)
            except:
                with tempfile.NamedTemporaryFile(suffix=".cif") as temp_cif:
                    resi = requests.get(
                        f"https://files.rcsb.org/download/{list(set(ll[1]['structure']))[0]}.cif"
                    )
                    if resi.status_code != 200:
                        print("404 " + list(set(ll[1]["structure"]))[0])

                    temp_cif.write(resi.content)
                    struct = MMCIFParser(QUIET=True).get_structure(
                        "str", temp_cif.name
                    )[0]
                    io = MMCIFIO()
                    io.set_structure(struct)
                    io.save(
                        temp_cif.name,
                        ChainsSelect(list(set(ll[1]["ture_chain"]))),
                    )
                    cmd.reinitialize()
                    cmd.load(
                        temp_cif.name,
                        object="model_",
                    )
                    cmd.save(ref_path)
                structure_fixer(ref_path)

            for id, dd in ll[1].iterrows():
                cmd.reinitialize()
                file = (
                    f"{directory_home}rnaquanet/data/00_reference/lociparse_train/{source}/"
                    + dd["common_structure"]
                    + "_"
                    + str(dd["model"])
                    + "/model.pdb"
                )
                structure_fixer(file)

                cmd.load(
                    file,
                    object="model_",
                )
                cmd.load(ref_path, object="reference_")
                try:
                    print(f"{dd['common_structure']}")
                    rmsd = cmd.pair_fit(f"model_", "reference_")
                    print(f"{rmsd} {dd['common_structure']}")
                    _models_df.loc[
                        id,
                        "RMSD",
                    ] = rmsd
                except:
                    ref = PDBParser().get_structure("str", ref_path)[0]
                    ref = ref[[i.id for i in ref][0]]
                    model = PDBParser().get_structure("str", file)[0]
                    res_ref = [i.get_resname() for i in ref]
                    _ref = [i for i in ref]
                    model = model[[i.id for i in model][0]]
                    res_mod = [i.get_resname() for i in model]
                    _mod = [i for i in model]

                    persist_ref_residue = []
                    persist_mod_residue = []

                    # clearing structure with given sequence from pandas row, reference strucure
                    aligment = aligner.align(
                        Seq("".join(res_ref)), Seq(str(dd["sequence"]))
                    )[0]

                    alignment_ref, alignment_mod = aligment
                    for alig_m, alig_r, alig_m_i, alig_r_i in zip(
                        alignment_mod,
                        alignment_ref,
                        aligment.indices[1],
                        aligment.indices[0],
                    ):
                        if alig_m != "-" and alig_m == alig_r and alig_r_i > -1:
                            persist_ref_residue.append(_ref[alig_r_i])

                    # clearing structure with given sequence from pandas row, model strucure
                    aligment = aligner.align(
                        Seq(str(dd["sequence"])), Seq("".join(res_mod))
                    )[0]
                    alignment_ref, alignment_mod = aligment

                    for alig_m, alig_r, alig_m_i, alig_r_i in zip(
                        alignment_mod,
                        alignment_ref,
                        aligment.indices[1],
                        aligment.indices[0],
                    ):
                        if alig_r != "-" and alig_m == alig_r and alig_m_i > -1:
                            persist_mod_residue.append(_mod[alig_m_i])

                    print(
                        f"{dd['common_structure']} {dd['model']}\n{len(res_ref)} {''.join(res_ref)}\n{len(res_mod)} {''.join(res_mod)}"
                    )

                    io = PDBIO()
                    io.set_structure(ref)
                    io.save(
                        ref_path,
                        ResidueSelect(persist_ref_residue),
                    )
                    io = PDBIO()
                    io.set_structure(model)
                    io.save(
                        file,
                        ResidueSelect(persist_mod_residue),
                    )

                    # structure_fixer(file)
                    # structure_fixer(ref_path)

                    ref = PDBParser().get_structure("str", ref_path)[0]
                    ref = ref[[i.id for i in ref][0]]
                    model = PDBParser().get_structure("str", file)[0]
                    model = model[[i.id for i in model][0]]
                    _ref = [i for i in ref]
                    _mod = [i for i in model]

                    persist_ref_residue = []
                    persist_mod_residue = []

                    # clearing structure to achieve both consistency
                    aligment = aligner.align(
                        Seq("".join([i.get_resname() for i in ref])),
                        Seq("".join([i.get_resname() for i in model])),
                    )[0]

                    alignment_ref, alignment_mod = aligment

                    for alig_m, alig_r, alig_m_i, alig_r_i in zip(
                        alignment_mod,
                        alignment_ref,
                        aligment.indices[1],
                        aligment.indices[0],
                    ):
                        if alig_r != "-" and alig_m == alig_r and alig_m_i > -1:
                            persist_mod_residue.append(_mod[alig_m_i])
                        if alig_m != "-" and alig_m == alig_r and alig_r_i > -1:
                            persist_ref_residue.append(_ref[alig_r_i])

                    io = PDBIO()
                    io.set_structure(ref)
                    io.save(
                        ref_path,
                        ResidueSelect(persist_ref_residue),
                    )
                    io = PDBIO()
                    io.set_structure(model)
                    io.save(
                        file,
                        ResidueSelect(persist_mod_residue),
                    )

                    cmd.reinitialize()
                    cmd.load(
                        file,
                        object="model_",
                    )
                    cmd.load(ref_path, object="reference_")

                    # print(
                    #     f"{dd['common_structure']}\n{''.join(res_ref)} {len(res_ref)}\n{alignment_ref}\n{alignment_mod}\n{''.join(res_mod)} {len(res_mod)}\n"
                    # )
                    _models_df.loc[
                        id,
                        "aligment",
                    ] = str(aligment)
                    rmsd = cmd.pair_fit(f"model_", "reference_")

                    print(f"{rmsd} {dd['common_structure']}")
                    _models_df.loc[
                        id,
                        "RMSD",
                    ] = rmsd

            _models_df.to_csv(
                models_df_path,
                index=False,
            )
    return rmsd_models, structure_fixer


@app.cell
def _(mo):
    mo.md(r"""### Split processing into chunks to perform multiprocessing""")
    return


@app.cell
def _(pd, rmsd_models):
    from multiprocessing import Pool
    val =pd.read_csv('/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/val_f.csv')
    # a=list(val.loc[val["RMSD"] == 999.0].groupby(
    #             "common_structure"
    # ))
    # chksize=len(a)//20+1
    # splited = [(e,a[i:i + chksize]) for e,i in enumerate(range(0, len(a), chksize))]
    # del a
    # for index,li in splited:
    #     descriptions = [i[0] for i in splited[index][1]]
    #     val.loc[val['common_structure'].isin(descriptions)].to_csv(f"{directory_home}rnaquanet/data/00_reference/lociparse_train/val_f_{index}.csv",index=None)
    paths=[f'/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/val_f_{i}.csv' for i in range(20)]
    processes = []
    # with Pool(20) as p:
    #     print(p.map(rmsd_models, [('val',p) for p in paths]))
    rmsd_models(('val',paths[8]))


    return Pool, paths, processes, val


@app.cell
def _(mo):
    mo.md(r"""### Testing section, achieving sequence and testing aligment""")
    return


@app.cell
def _(
    ChainsSelect,
    MMCIFIO,
    MMCIFParser,
    PDBIO,
    PDBParser,
    cmd,
    ll,
    requests,
    structure_fixer,
    tempfile,
):
    ref_path2 = "/tmp/test.pdb"
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as temp_cif:
            resi = requests.get(f"https://files.rcsb.org/download/4K27.pdb")
            if resi.status_code != 200:
                print("404 " + list(set(ll[1]["structure"]))[0])

            temp_cif.write(resi.content)
            # structure_fixer(temp_cif.name)
            struct = PDBParser(QUIET=True).get_structure("str", temp_cif.name)[0]
            io = PDBIO()
            io.set_structure(struct)
            io.save(
                temp_cif.name,
                ChainsSelect(["U"]),
            )
            cmd.reinitialize()
            cmd.load(
                temp_cif.name,
                object="model_",
            )
            cmd.save(ref_path2)
    except:
        print("exception")
        with tempfile.NamedTemporaryFile(suffix=".cif") as temp_cif:
            resi = requests.get(f"https://files.rcsb.org/download/4K27.cif")
            if resi.status_code != 200:
                print("404 " + list(set(ll[1]["structure"]))[0])

            temp_cif.write(resi.content)

            struct = MMCIFParser(QUIET=True).get_structure("str", temp_cif.name)[0]
            io = MMCIFIO()
            io.set_structure(struct)
            io.save(
                temp_cif.name,
                ChainsSelect(["U"]),
            )
            cmd.reinitialize()
            cmd.load(
                temp_cif.name,
                object="model_",
            )
            cmd.save(ref_path2)
    structure_fixer(ref_path2)
    ref = PDBParser(QUIET=True).get_structure("str", ref_path2)[0]
    ref = ref[[i.id for i in ref][0]]
    "".join([i.get_resname() for i in ref])
    return io, ref, ref_path2, resi, struct, temp_cif


@app.cell
def _(Seq, aligner):
    aligner.open_gap_score = -12
    aligner.end_gap_score = 0
    print(
        aligner.align(
            Seq(
                "GUCACGCACAGAGCAAACCAUUCGAAAGAGUGGGACGCAAAGCCUCCGGCCUAAACCAUUGCACUCCGGUAGGUAGCGGGGUUAUCGAUG"
            ),
            Seq("GGUCACGCACAGAGCAAACCAUUCGAAAGAGGUCACGCACAGAGCAAACCAUUCGAAAGAGU"),
        )[0]
    )
    return


@app.cell
def _(pd):
    inde=19
    train_f_1=pd.read_csv(f'/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/val_f_{inde}.csv')
    train_f_1.loc[(train_f_1['RMSD']==999.0) | (train_f_1['aligment'].str.contains('target')),:]
    return inde, train_f_1


@app.cell
def _():
    # train_f_1.loc[train_f_1['common_structure']=='3MUT_1','RMSD']=999.0
    # train_f_1.loc[train_f_1['common_structure']=='3MUT_1','aligment']=''
    # train_f_1.loc[train_f_1['common_structure']=='3MUT_1','sequence']='GUCACGCACAGAGCAAACCAUUCGAAAGAGUGGGACGCAAAGCCUCCGGCCUAAA'
    return


@app.cell
def _(inde, train_f_1):
    train_f_1.to_csv(f'/home/adamczykb/rnaquanet/data/00_reference/lociparse_train/val_f_{inde}.csv',index=False)
    return


@app.cell
def _(models_df, plt, tqdm):
    import altair as alt
    v=[]
    # fig, ax = plt.subplots(len(models_df.loc[models_df['RMSD']<=15.0].groupby("common_structure")), sharex=True, sharey=True,figsize=(10, 20))
    iter=0
    for r,row in tqdm(models_df.loc[models_df['RMSD']<=15.0].groupby("common_structure")):
        fig,ax = plt.subplots()
        ax.hist(row['RMSD'])
        ax.set_title(r)
        ax.set_xlim((0, 15))
        plt.show()
        iter+=1

    # mo.vstack([ax])
    return alt, ax, fig, iter, r, row, v


@app.cell
def _(models_df):
    models_df['RMSD'].hist(range=[0, 20],bins=20)
    return


@app.cell
def _(models_df):
    import seaborn as sns
    sns.jointplot(data = models_df,
                  x = "length",
                  y = "RMSD",
                  marginal_ticks = True,s=1.4
                 )
    return (sns,)


@app.cell
def _(models_df, sns):
    sns.jointplot(data = models_df,
                  x = "length",
                  y = "RMSD",
                  marginal_ticks = True,s=1.4,ylim=(0,20)
                 )
    return


@app.cell
def _(models_df, sns):
    sns.jointplot(data = models_df.groupby('common_structure').agg({'length':'max','RMSD':'mean'}),
                  x = "length",
                  y = "RMSD",
                  marginal_ticks = True,s=1.4,ylim=(0,20)
                 )
    return


@app.cell
def _(models_df, sns):
    sns.jointplot(data = models_df.groupby('common_structure').agg({'length':'max','RMSD':'median'}),
                  x = "length",
                  y = "RMSD",
                  marginal_ticks = True,s=1.4,ylim=(0,20)
                 )
    return


if __name__ == "__main__":
    app.run()
