import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""#Farfar2 model generation pipeline""")
    return


@app.cell
def _():
    import subprocess
    import multiprocessing
    import pandas as pd
    import matplotlib.pyplot as plt
    return multiprocessing, pd, plt, subprocess


@app.cell
def _(pd):
    structure_list_=pd.read_csv('/home/adamczykb/rnaquanet/data/00_reference/selected_structures_for_dataseto.csv')
    return (structure_list_,)


@app.cell
def _(structure_list_):
    structure_list_['']
    return


@app.cell
def _(plt):
    res = [4.180617809295654,
     7.147128105163574,
     7.750968933105469,
     6.574105739593506,
     7.176124572753906,
     7.052803993225098,
     6.906412124633789,
     6.159549713134766,
     7.603024482727051,
     7.556703567504883,
     8.581474304199219,
     7.175538063049316,
     7.872018337249756,
     7.504454135894775,
     7.465034008026123,
     7.05313777923584,
     8.487544059753418,
     6.869085788726807,
     8.592341423034668,
     6.774190425872803]
    plt.hist(res)

    return (res,)


if __name__ == "__main__":
    app.run()
