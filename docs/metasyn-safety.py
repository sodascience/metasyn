import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Use marimo to create this post
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How privacy-safe is metasyn?

    We market metasyn as a safe, transparent, and auditable way of producing synthetic data. But how safe is metasyn, really? In this document, we show some of the ways in which metasyn protects against various types of privacy risks. We here reprint three common disclosure risks, along with their excellent explanation from [the `sdcMicro` documentation](https://sdcpractice.readthedocs.io/en/latest/measure_risk.html#types-of-disclosure)

    - __Identity disclosure__, which occurs if the intruder associates a known individual with a released data record. For example, the intruder links a released data record with external information, or identifies a respondent with extreme data values. In this case, an intruder can exploit a small subset of variables to make the linkage, and once the linkage is successful, the intruder has access to all other information in the released data related to the specific respondent.
    - __Attribute disclosure__, which occurs if the intruder is able to determine some new characteristics of an individual based on the information available in the released data. Attribute disclosure occurs if a respondent is correctly re-identified and the dataset contains variables containing information that was previously unknown to the intruder. Attribute disclosure can also occur without identity disclosure. For example, if a hospital publishes data showing that all female patients aged 56 to 60 have cancer, an intruder then knows the medical condition of any female patient aged 56 to 60 in the dataset without having to identify the specific individual.
    - __Inferential disclosure__, which occurs if the intruder is able to determine the value of some characteristic of an individual more accurately with the released data than would otherwise have been possible. For example, with a highly predictive regression model, an intruder may be able to infer a respondent’s sensitive income information using attributes recorded in the data, leading to inferential disclosure.

    To make our case, we use the `metasyn` package with its `metasyn-disclosure` privacy plugin, and we will use the built-in example hospital dataset to assess our disclosure risks, which we will manipulate using the `polars` library.
    """)
    return


@app.cell
def _():
    import polars as pl
    import plotnine as p9
    import metasyn as ms
    from metasyncontrib.disclosure import DisclosurePrivacy

    df = ms.demo_dataframe("hospital")
    df
    return df, ms, p9, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This example dataset has 18 patients with their characteristics such as age, admission date / time, disease type, and the number of hours in the hospital. An example analysis could be around figuring out what the impact of the different disease types is on the hospital in terms of the number of hours they keep the room occupied, on average:
    """)
    return


@app.cell(hide_code=True)
def _(df, p9, pl):
    tab = (
        df.group_by(pl.col.type)
        .agg(
            n=pl.len(),
            mean_hours=pl.col.hours_in_room.mean().round(2),
            std_err=(pl.col.hours_in_room.std() / pl.len().sqrt()).round(2),
        )
        .sort(pl.col.type)
    )


    plt = (
        p9.ggplot(
            tab,
            p9.aes(
                x="type",
                y="mean_hours",
                ymin="mean_hours + std_err",
                ymax="mean_hours - std_err",
            ),
        )
        + p9.geom_pointrange()
        + p9.theme_light()
        + p9.labs(
            title="Hours in hospital by disease type",
            x="",
            y="Hours in hospital",
            caption="(Error bars indicate standard error of the mean)",
        )
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating synthetic data

    To generate synthetic data with metasyn, we first create a `MetaFrame`, which contains all the information to `synthesize()` new data in the same format as the original data.
    """)
    return


@app.cell
def _(df, ms):
    mf = ms.MetaFrame.fit_dataframe(df)
    return (mf,)


@app.cell
def _(mf):
    df_synth = mf.synthesize()
    df_synth
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identity disclosure

    According to [statistics canada](https://www150.statcan.gc.ca/n1/en/pub/11-522-x/2025001/article/00016-eng.pdf?st=Pusgy49_),

    > identity disclosure occurs when an intruder associates a known individual with a released data record

    Synthetic data has low identity disclosure risk in general, because there is no direct one-to-one link between any released data record (i.e., a row in our synthetic dataframe) and the original data (i.e., a row in our sensitive dataframe). However, we can look at what information is stored about the `patient_id` just to be sure about what information is transferred about individuals identifiers:
    """)
    return


@app.cell
def _(mf):
    print(mf["patient_id"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As shown here, the only information about patient identifiers encoded in the `MetaFrame` is their _structure_, not their value. This structure is encoded as a [regular expression](https://en.wikipedia.org/wiki/Regular_expression), which metasyn automatically deduces via the `regexmodel` package.

    In short, the regex `[A-B][0-9]{4}X[0-9]` means: start with the letter A or B (`[A-B]`), then have 4 numbers (`[0-9]{4}`), then put an X, and end with another number (`[0-9]`). Metasyn can then generate new identifiers following this structure like so:
    """)
    return


@app.cell
def _(mf):
    mf["patient_id"].draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the values of identifiers are not stored or used at all, these do not correspond to real identifiers.

    Another risk here is related to membership inference; using outliers we can infer  For example, if we know this data is from a hospital
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attribute disclosure

    Another
    """)
    return


if __name__ == "__main__":
    app.run()
