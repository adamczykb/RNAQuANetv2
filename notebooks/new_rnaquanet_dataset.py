import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Generation of new RNAQuANet dataset""")
    return


@app.cell
def _():
    import pandas as pd
    from Bio.PDB import PDBParser
    import os
    from pymol import cmd
    from Bio.PDB import MMCIFParser
    import Bio
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    return Bio, MMCIFParser, PDBParser, cmd, os, pd, plt, tqdm


@app.cell
def _(mo):
    mo.md(r"""## Clear structures from nonRNA elements""")
    return


@app.cell
def _(os, pd):
    structure_list =  pd.read_csv('/home/adamczykb/rnaquanet/data/00_reference/BSGU__R__All__All__3_0__txt.txt')['structure'].unique().tolist()
    structure_list_o = os.listdir(
        "/home/adamczykb/rnaquanet/data/00_reference/BGSU__R__All__All__4_0__cif_3_378"
    )
    structure_list = [x.split(".")[0] for x in structure_list_o]
    structure_list = [x.replace("Chain_id", "") for x in structure_list]
    structure_list = [stri.split("_") for stri in  structure_list]
    structure_list = pd.DataFrame(structure_list)[[0, 2]]
    structure_list.rename(columns={0: "pdb_id", 2: "chain_id"}, inplace=True)
    structure_list['path']= "/home/adamczykb/rnaquanet/data/00_reference/BGSU__R__All__All__4_0__cif_3_378/"+pd.Series(structure_list_o)

    structure_list['res']=0.0
    structure_list['res2']=0.0
    structure_list['size']=0.0
    structure_list['sequence']=''
    structure_list['struct_keywords']=''
    structure_list['type']=''

    import json
    from numpy import inf
    import requests



    r = requests.post(
        "https://data.rcsb.org/graphql",
        json={
            "query": """query Elements($pdb_ids:[String!]!) {
                            entries(entry_ids: $pdb_ids) {
                                refine{
                                ls_d_res_high
                                }
                                em_3d_reconstruction {
                                resolution
                                }
                                rcsb_entry_info {
                                    polymer_entity_count_protein
                                }
                                entry {
                                id,

                                },
                                struct_keywords {
                                    pdbx_keywords

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
            "variables": {"pdb_ids": list(structure_list['pdb_id'].unique().tolist())},
        },
    )
    for entry in json.loads(r.text)["data"]["entries"]:
        structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'res'] = (entry["refine"][0]["ls_d_res_high"] if entry["refine"] else inf)
        structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'res2'] = (entry["em_3d_reconstruction"][0]["resolution"] if entry["em_3d_reconstruction"] else inf)
        structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'type'] = entry["struct_keywords"]['pdbx_keywords']
        if (
            "protein" in entry["struct_keywords"]["text"].lower()
            or "ribosom" in entry["struct_keywords"]["text"].lower()
            or "complex" in entry["struct_keywords"]["text"].lower()
            or entry["rcsb_entry_info"]["polymer_entity_count_protein"] > 0
        ):
            structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'struct_keywords'] = 'complex'


        elif "dna" in entry["struct_keywords"]["text"].lower():
            structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'struct_keywords'] = 'hybrid'


        elif "rna" in entry["struct_keywords"]["text"].lower():
            structure_list.loc[structure_list['pdb_id']==entry["entry"]["id"].upper(),'struct_keywords'] = 'solo'


        else:
            print(f"unknown: {entry}")
    return entry, inf, json, r, requests, structure_list, structure_list_o


@app.cell
def _(Bio, inf, structure_list):
    structure_list['resolution (A)'] = structure_list['res']
    structure_list.loc[structure_list['resolution (A)']==inf,'resolution (A)'] = structure_list.loc[structure_list['resolution (A)']==inf,'res2']
    structure_list.drop(columns=['res','res2'], inplace=True)
    structure_list.drop(structure_list.loc[structure_list['resolution (A)']==inf].index, inplace=True)
    structure_list.drop_duplicates(subset=['pdb_id','chain_id'], inplace=True)
    class RNA(Bio.PDB.Select):
        RNA_DICT_FULL = ['A', 'C', 'G', 'U','DU', 'A23', 'A2L', 'A2M', 'A39', 'A3P', 'A44', 'A5O', 'A6A', 'A7E', 'A9Z', 'ADI', 'ADP', 'AET', 'AMD', 'AMO', 'AP7', 'AVC', 'MA6', 'MAD', 'MGQ', 'MIA', 'MTU', 'M7A', '26A', '2MA', '6IA', '6MA', '6MC', '6MP', '6MT', '6MZ', '6NW', 'F3N', 'N79', 'RIA', 'V3L', 'ZAD', '31H', '31M', '7AT', 'O2Z', 'SRA', '00A', '45A', '8AN', 'LCA', 'P5P', 'PPU', 'PR5', 'PU', 'T6A', 'TBN', 'TXD', 'TXP', '12A', '1MA', '5FA', 'A6G', 'E6G', 'E7G', 'EQ4', 'IG', 'IMP', 'M2G', 'MGT', 'MGV', 'MHG', 'QUO', 'YG', 'YYG', '23G', '2EG', '2MG', '2SG', 'B8K', 'B8W', 'B9B', 'BGH', 'N6G', 'RFJ', 'ZGU', '7MG', 'CG1', 'G1G', 'G25', 'G2L', 'G46', 'G48', 'G7M', 'GAO', 'GDO', 'GDP', 'GH3', 'GNG', 'GOM', 'GRB', 'GTP', 'KAG', 'KAK', 'O2G', 'OMG', '8AA', '8OS', 'LG', 'PGP', 'P7G', 'TPG', 'TG', 'XTS', '102', '18M', '1MG', 'A5M', 'A6C', 'E3C', 'IC', 'M4C', 'M5M', '6OO', 'B8Q', 'B8T', 'B9H', 'JMH', 'N5M', 'RPC', 'RSP', 'RSQ', 'ZBC', 'ZCY', '73W', 'C25', 'C2L', 'C31', 'C43', 'C5L', 'CBV', 'CCC', 'CH', 'CSF', 'OMC', 'S4C', '4OC', 'LC', 'LHH', 'LV2', 'PMT', 'TC', '10C', '1SC', '5HM', '5IC', '5MC', 'A6U', 'IU', 'I4U', 'MEP', 'MNU', 'U25', 'U2L', 'U2P', 'U31', 'U34', 'U36', 'U37', 'U8U', 'UAR', 'UBB', 'UBD', 'UD5', 'UPV', 'UR3', 'URD', 'US5', 'UZR', 'UMO', 'U23', '2AU', '2MU', '2OM', 'B8H', 'FHU', 'FNU', 'F2T', 'RUS', 'ZBU', '3AU', '3ME', '3MU', '3TD', '70U', '75B', 'CNU', 'OMU', 'ONE', 'S4U', 'SSU', 'SUR', '4SU', '85Y', 'DHU', 'H2U', 'LHU', 'PSU', 'PYO', 'P4U', 'T31', '125', '126', '127', '1RN', '5BU', '5FU', '5MU', '9QV', '5GP']

        def __init__(self):
           pass

        def accept_residue(self, residue):
            return residue.get_resname() in self.RNA_DICT_FULL
    return (RNA,)


@app.cell
def _(MMCIFParser, RNA, structure_list, tqdm):
    from Bio.PDB.mmcifio import MMCIFIO

    for structure in tqdm(structure_list['path'].tolist()):
        try:
            s=MMCIFParser().get_structure('str',structure)
            io = MMCIFIO()
            io.set_structure(s)
            io.save(
                structure,
                RNA(),
            )
            structure_list.loc[structure_list.path == structure,'size']=len(list(MMCIFParser().get_structure('str',structure).get_residues()))
        except:
            structure_list.loc[structure_list.path == structure,'size']=0
            print(structure)
    return MMCIFIO, io, s, structure


@app.cell
def _(structure_list):
    structure_list_cleared = structure_list[(structure_list['size']>=20)&(structure_list['size']<500)]
    structure_list_cleared.to_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataset.csv',index=False)
    return (structure_list_cleared,)


@app.cell
def _(MMCIFParser, structure_list_cleared):
    for h, struct in structure_list_cleared.iterrows():
        structure_cif = MMCIFParser().get_structure("str", struct["path"])[0]
        chains_ascending = sorted(struct["chain_id"].split("-"))
        sequences=[]
        for chain in chains_ascending:
            if chain in structure_cif:
                sequences.append(''.join([r.resname for r in structure_cif[chain].get_residues()]))
            else:
                chains_ascending.remove(chain)
        if len(chains_ascending)==0 or len(sequences)==0:
            structure_list_cleared.drop(axis=1,index=h,inplace=True)
            print(f"{h} {struct}")
            continue
        structure_list_cleared.loc[h,"chain_id"] = "-".join(chains_ascending)
        structure_list_cleared.loc[h,"sequence"] = ";".join(sequences)
    return chain, chains_ascending, h, sequences, struct, structure_cif


@app.cell
def _(structure_list_cleared):
    structure_list_cleared.to_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataset.csv',index=False)
    return


@app.cell
def _(structure_list_cleared):
    import shutil
    for index, row in structure_list_cleared.iterrows():
        try:
            shutil.copy(row['path'], f'/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/{row['path'].split('/')[-1]}')
        except shutil.SameFileError:
            pass
    return index, row, shutil


@app.cell
def _(mo, structure_list_cleared):
    mo.ui.table(structure_list_cleared)
    return


@app.cell
def _(structure_list_cleared):
    for index_, row_ in structure_list_cleared.iterrows():
        structure_list_cleared.loc[index_,'file']=row_['path'].split('/')[-1]

    structure_list_cleared.drop(columns='path',axis=0,inplace=True)
    structure_list_cleared.to_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataset.csv',index=False)
    structure_list_cleared['boltz_prediciton_rmsd']=9999.0
    return index_, row_


@app.cell
def _():
    ## Preparing structures for Boltz
    return


@app.cell
def _(cmd, structure_list_cleared, tqdm):
    import subprocess
    structure_list_cleared['chain_id']=structure_list_cleared['chain_id'].str.replace('Chain_id','')
    structure_list_cleared['file']=structure_list_cleared['file'].str.replace('Chain_id','')
    for index__, row__ in tqdm(structure_list_cleared.iterrows()):
        try:
            cmd.reinitialize()
            cmd.load('/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/'+row__['file'],object='reference')
            cmd.save('/home/adamczykb/rnaquanet/data/00_reference/fastas/'+row__['file'].split('.')[0].replace('Chain_id','')+'.fasta')
            lines=[]
            with open('/home/adamczykb/rnaquanet/data/00_reference/fastas/'+row__['file'].split('.')[0].replace('Chain_id','')+'.fasta','r') as f:
                lines=f.readlines()
            for l in range(len(lines)):
                if '>reference_' in lines[l]:
                    lines[l]=lines[l].replace('>reference_','>').replace('\n','')+'|rna \n'
            with open('/home/adamczykb/rnaquanet/data/00_reference/fastas/'+row__['file'].split('.')[0].replace('Chain_id','')+'.fasta','w') as f:
                f.seek(0)
                f.writelines(lines)

            # process = subprocess.Popen(
            #     ("docker run -it --gpus all -v /home/adamczykb/rnaquanet/data/00_reference:/data boltz:latest boltz predict "+'/data/fastas/'+i['file'].split('.')[0]+'.fasta'+ ' --accelerator gpu --output_format mmcif --out_dir '+ "/data/boltz_result/"+i['file'].split('.')[0]).split()
            # )
            # display(process.communicate())
            cmd.reinitialize()
            cmd.load('/home/adamczykb/rnaquanet/data/00_reference/boltz_result/boltz_results_fastas/predictions/'+row__['file'].split('.')[0]+'/'+row__['file'].split('.')[0]+'_model_0.cif',object='boltz')
            cmd.load('/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/'+row__['file'],object='reference')
            structure_list_cleared.loc[index__,'boltz_prediciton_rmsd']=cmd.align(f"boltz", "reference", cycles=0)[0]
        except:
            structure_list_cleared.drop(axis=1,index=index__,inplace=True)
    return f, index__, l, lines, row__, subprocess


@app.cell
def _(cmd, i, structure_list_cleared, tqdm):
    for index___, row___ in tqdm(structure_list_cleared.iterrows()):
        try:
            cmd.reinitialize()
            cmd.load('/home/adamczykb/rnaquanet/data/00_reference/boltz_result/boltz_results_fastas/predictions/'+row___['file'].split('.')[0].replace('Chain_id','')+'/'+row___['file'].replace('Chain_id','').split('.')[0]+'_model_0.cif',object='boltz')
            cmd.load('/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/'+row___['file'],object='reference')
            structure_list_cleared.loc[index___,'boltz_prediciton_rmsd']=cmd.align(f"boltz", "reference", cycles=0)[0]
        except:
            print(f"{row___} "+'/home/adamczykb/rnaquanet/data/00_reference/boltz_result/boltz_results_fastas/predictions/'+row___['file'].split('.')[0].replace('Chain_id','')+'/'+i['file'].replace('Chain_id','').split('.')[0]+'_model_0.cif')
    structure_list_cleared.drop(index=structure_list_cleared.loc[structure_list_cleared['boltz_prediciton_rmsd']==9999.0].index,inplace=True)
    structure_list_cleared.to_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataseto.csv',index=False)
    return index___, row___


@app.cell
def _(plt, structure_list_cleared):
    plt.scatter(
        structure_list_cleared[["size", "boltz_prediciton_rmsd"]].to_numpy()[:, 0],
        structure_list_cleared[["size", "boltz_prediciton_rmsd"]].to_numpy()[:, 1],
        s=0.1,
    )
    plt.xlabel('structure size')
    plt.ylabel('RMSD boltz')
    return


@app.cell
def _(plt, structure_list_cleared):
    plt.scatter(
        structure_list_cleared.loc[
            structure_list_cleared["boltz_prediciton_rmsd"] < 10.0,
            ["size", "boltz_prediciton_rmsd"],
        ].to_numpy()[:, 0],
        structure_list_cleared.loc[structure_list_cleared['boltz_prediciton_rmsd']<10.0,['size','boltz_prediciton_rmsd']].to_numpy()[:,1],
        s=0.1,
    )
    plt.xlabel('structure size')
    plt.ylabel('RMSD boltz')
    return


@app.cell
def _(plt, structure_list_cleared):
    structure_list_cleared.loc[
            structure_list_cleared["boltz_prediciton_rmsd"] < 10.0,"boltz_prediciton_rmsd"
    ].hist()
    plt.xlabel('RMSD boltz')
    return


@app.cell
def _(mo):
    mo.md(r"""## Boltz prediction for 8G60_1_At""")
    return


@app.cell
def _(cmd, tqdm):
    import random
    import subprocess
    import shutil
    results=[]
    for i in tqdm(range(100)):
        process = subprocess.Popen(f'docker run --gpus all -v /home/adamczykb/rnaquanet/data/00_reference:/data boltz:latest boltz predict /data/fastas/8G60_1_At.fasta --accelerator gpu --output_format mmcif --out_dir /data/boltz_resultt/ --seed {int(random.random()*10000000000)} --override'.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        while True:
            output = process.stdout.read()
            outputerr = process.stderr.read()
            if len(outputerr)>0:
                print(outputerr.decode())
            if process.poll() is not None:
                break
            if output:
                print(outputerr.decode().strip(),end='')

        cmd.reinitialize()
        cmd.load('/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/boltz_results_8G60_1_At/predictions/8G60_1_At/8G60_1_At_model_0.cif',object='boltz')
        cmd.load('/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/'+'8G60_1_At.cif',object='reference')
        results.append([i,cmd.align(f"boltz", "reference", cycles=0)[0]])
        print(f"RMSD: {results[-1]}")
        shutil.copy('/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/boltz_results_8G60_1_At/predictions/8G60_1_At/8G60_1_At_model_0.cif',f'/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/8G60_1_At/{i}.cif')
        shutil.rmtree('/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/boltz_results_8G60_1_At')
    
    return i, output, outputerr, process, random, results, shutil, subprocess


@app.cell
def _():
    ## Boltz model matrix for 8G60_1_At
    return


@app.cell
def _(pd):
    structure_list_=pd.read_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataseto.csv')
    return (structure_list_,)


@app.cell
def _(cmd, plt):
    import numpy as np
    import seaborn as sns

    table = np.zeros((18,18))
    for _i in range(18):
        for _j in range(_i+1,18):
            cmd.reinitialize()
            cmd.load(f'/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/8G60_1_At/{_i}.cif',object='boltz')
            cmd.load(f'/home/adamczykb/rnaquanet/data/00_reference/boltz_resultt/8G60_1_At/{_j}.cif',object='reference')
            _r=cmd.align(f"boltz", "reference", cycles=0)[0]
            table[_i,_j]=_r
            table[_j,_i]=_r
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(table,ax = ax, annot=True,)
    return ax, fig, np, sns, table


app._unparsable_cell(
    r"""
    from openmm.app import *
    from openmm import *
    from openmm.unit import *

    # Load a mmCIF file
    structure = PDBxFile(
        \"/home/adamczykb/rnaquanet/data/00_reference/newrnaquanet/\" + \"3K1V_1_A.cif\"
    )

    # Create a force field
    forcefield = ForceField('amber14-all.xml','amber14/tip3pfb.xml')

    platform = Platform.getPlatform(\"CPU\")
    modeller = Modeller(structure.topology, structure.positions)
    modeller.addHydrogens()
    modeller.addExtraParticles(forcefield)
    # modeller.addHydrogens(forcefield)
    # Create a system from the topology with force field parameters
    system = forcefield.createSystem(
        modeller.topology,
        # nonbondedMethod=PME,
        # nonbondedCutoff=1 * nanometer,
        constraints=HBonds,
    )
    # Create an integrator
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)

    mergedTopology = modeller.topology
    mergedPositions = modeller.positions

    # Set up the simulation
    simulation = Simulation(mergedTopology, system, integrator)
    simulation.context.setPositions(mergedPositions)

    # Energy minimize
    simulation.minimizeEnergy()

    # Add reporters for output

    # simulation.reporters.append(
    #     StateDataReporter(
    #         \"output.txt\", 1000, step=True, potentialEnergy=True, temperature=True
    #     )
    # )

    # Run the simulation
    simulation.step(10)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    with open(\"final_state.cif\", mode=\"w\") as file:
        PDBxFile.writeFile(simulation.topology, state.getPositions(), file)


    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
