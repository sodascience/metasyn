import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="metasyn - synthesize")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convert your GMF file to a synthetic data file
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    from metasyn import MetaFrame
    from metasyn.file import CsvFileInterface, ExcelFileInterface, SavFileInterface, StataFileInterface
    from pyreadstat import write_sav, write_dta
    from io import BytesIO
    import re
    from pathlib import Path

    return (
        BytesIO,
        CsvFileInterface,
        ExcelFileInterface,
        MetaFrame,
        Path,
        SavFileInterface,
        StataFileInterface,
        json,
        mo,
        re,
        write_dta,
        write_sav,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Upload your GMF (.json) file below
    """)
    return


@app.cell
def _(mo):
    file_ui = mo.ui.file(kind="area")
    file_ui
    return (file_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview of the Synthetic data
    """)
    return


@app.cell
def _(MetaFrame, file_ui, json):
    data = file_ui.contents()
    mf = MetaFrame.load_json(json.loads(data.decode("utf-8")))
    mf.synthesize(10, progress_bar=False)
    return (mf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate synthetic data with the file format in the file (if available)

    It depends on the creators of the GMF file whether this information is available. If it is, you can generate a file that is as similar as possible to the original below, otherwise the button for download will not appear, but you can select the file format in the next section manually.
    """)
    return


@app.cell
def _(BytesIO, Path, re):
    def unescape(regex_str: str) -> str:
        """Remove slashes that are in the serialized string."""
        return re.sub(r'\\\\(.)', r'\1', regex_str)

    def gen_file(mf, file_format):
        handler = BytesIO()
        mf.write_synthetic(handler, file_format=file_format, progress_bar=False)
        return handler

    from tempfile import NamedTemporaryFile

    def load_prs_data(mf, file_format, prs_func):
        with NamedTemporaryFile(delete_on_close=False) as fp:
            if Path(fp.name).is_file():
                Path(fp.name).unlink()
            mf.write_synthetic(fp.name, file_format=file_format, progress_bar=False)
            handler = BytesIO()
            with open(fp.name, "rb") as fp_handle:
                handler.write(fp_handle.read())
        handler.seek(0)
        return handler

    return gen_file, load_prs_data


@app.cell
def _(gen_file, mf, mo):
    elem = None
    if mf.file_format is not None:
        elem = mo.download(data=gen_file(mf, None), label="Generate data file", mimetype="text/csv")
    elem
    return


@app.cell
def _(mo):
    csv_sep = mo.ui.text(value=",")
    line_term = mo.ui.text(value="\\n")
    quote_char = mo.ui.text(value="\"")
    null_value = mo.ui.text(value="")
    encoding = mo.ui.text(value="utf-8")
    csv_dict = {"separator:": csv_sep, "line termination:": line_term, "quotation mark:": quote_char, "null value:": null_value, "encoding:": encoding}
    return csv_dict, csv_sep, encoding, line_term, null_value, quote_char


@app.cell
def _(mo):
    xls_work_sheet = mo.ui.text(value="Sheet1")
    return (xls_work_sheet,)


@app.cell
def _(
    CsvFileInterface,
    ExcelFileInterface,
    SavFileInterface,
    StataFileInterface,
    csv_sep,
    encoding,
    line_term,
    null_value,
    quote_char,
    xls_work_sheet,
):
    csv_file_format = CsvFileInterface({
            "separator": csv_sep.value,
            "quote_char": quote_char.value,
            "line_terminator": line_term.value.replace("\\n", "\n").replace("\\r", "\r"),
            "null_value": null_value.value,
            "encoding": encoding.value,
        }, "synthetic_data.csv")

    xls_file_format = ExcelFileInterface({"worksheet": xls_work_sheet.value}, "synthetic_data.xlsx")
    stata_file_format = StataFileInterface({}, "synthetic_data.dta")
    sav_file_format = SavFileInterface({}, "synthetic_data.sav")
    return csv_file_format, sav_file_format, stata_file_format, xls_file_format


@app.cell
def _(
    load_prs_data,
    mf,
    sav_file_format,
    stata_file_format,
    write_dta,
    write_sav,
):
    def load_sav():
        return load_prs_data(mf, sav_file_format, write_sav)

    def load_stata():
        return load_prs_data(mf, stata_file_format, write_dta)

    return load_sav, load_stata


@app.cell
def _(mo):
    csv_file_name = mo.ui.text(value="synthetic_data.csv")
    xls_file_name = mo.ui.text(value="synthetic_data.xls")
    stata_file_name = mo.ui.text(value="synthetic_data.dta")
    sav_file_name = mo.ui.text(value="synthetic_data.sav")
    return csv_file_name, sav_file_name, stata_file_name, xls_file_name


@app.cell
def _(csv_file_format, csv_file_name, gen_file, mf, mo):
    csv_download = mo.download(data=gen_file(mf, csv_file_format), filename=csv_file_name.value, mimetype="text/csv", label="Generate CSV")
    csv_gen = mo.hstack([mo.md("File name:").style({"width": "120px"}), csv_file_name], widths=[0,1])
    csv_click = mo.hstack([mo.md("").style({"width": "120px"}), csv_download], widths=[0,1])
    return csv_click, csv_gen


@app.cell
def _(gen_file, mf, mo, xls_file_format, xls_file_name, xls_work_sheet):
    xls_mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if xls_file_name.value.endswith("xlsx") else "application/vnd.ms-excel"
    xls_file_row = mo.hstack([mo.md("File name:").style({"width": "120px"}), xls_file_name], widths=[0,1])
    xls_download = mo.download(data=gen_file(mf, xls_file_format), filename=xls_file_name.value, mimetype=xls_mime_type, label="Generate Excel file")
    xls_click = mo.hstack([mo.md("").style({"width": "120px"}), xls_download], widths=[0,1])
    xls_sheet_row = mo.hstack([mo.md("Sheet name:").style({"width": "120px"}), xls_work_sheet], widths=[0, 1])
    return xls_click, xls_file_row, xls_sheet_row


@app.cell
def _(load_stata, mo, stata_file_name):
    stata_file_row = mo.hstack([mo.md("File name:").style({"width": "80px"}), stata_file_name], widths=[0,1])
    stata_download = mo.download(data=load_stata, filename=stata_file_name.value, mimetype="MimeType=application/x-stata-dta", label="Generate Stata file")
    stata_click = mo.hstack([mo.md("").style({"width": "80px"}), stata_download], widths=[0,1])
    return stata_click, stata_file_row


@app.cell
def _(load_sav, mo, sav_file_name):
    sav_file_row = mo.hstack([mo.md("File name:").style({"width": "80px"}), sav_file_name], widths=[0,1])
    sav_download = mo.download(data=load_sav, filename=sav_file_name.value, mimetype="MimeType=application/x-spss-sav", label="Generate Sav file")
    sav_click = mo.hstack([mo.md("").style({"width": "80px"}), sav_download], widths=[0,1])
    return sav_click, sav_file_row


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate synthetic data files in different formats
    """)
    return


@app.cell
def _(
    csv_click,
    csv_dict,
    csv_gen,
    mo,
    sav_click,
    sav_file_row,
    stata_click,
    stata_file_row,
    xls_click,
    xls_file_row,
    xls_sheet_row,
):
    mo.hstack((
        mo.vstack([mo.hstack([mo.md(label).style({"width": "120px"}), elem], widths=[0, 1]) for label, elem in csv_dict.items()] + [csv_gen, csv_click]),
        mo.vstack([xls_sheet_row, xls_file_row, xls_click]),
        mo.vstack([stata_file_row, stata_click]),
        mo.vstack([sav_file_row, sav_click])
        ),
        align="end"
    )
    return


if __name__ == "__main__":
    app.run()
