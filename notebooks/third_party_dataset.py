import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    return mo, os, pd


@app.cell
def _(os):
    dir=[i.split('.')[0] for i in os.listdir('/home/adamczykb/Downloads/training_npzs_light/npz_3633')]
    return (dir,)


@app.cell
def _(dir):
    dicts = dict()
    for i in dir:
        chains = dicts.get(i.split('_')[0],[])+i.split('_')[1:]
        dicts[i.split('_')[0]]=sorted(chains)
    return chains, dicts, i


@app.cell
def _(dicts):
    len(dicts.keys())
    return


@app.cell
def _(chains, dicts):
    normal_single_chain=[]
    for pdbid,chainss in dicts.items():
        for c in list(set(chains)):
            normal_single_chain.append(f"{pdbid}_{c}")
    return c, chainss, normal_single_chain, pdbid


@app.cell
def _(dicts):
    list(set(dicts))
    return


@app.cell
def _(dicts):
    sorted(dicts,key=lambda v: v.split('_')[0])
    return


if __name__ == "__main__":
    app.run()
