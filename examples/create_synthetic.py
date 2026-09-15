import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="metasyn")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metasyn - generate synthetic data
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from metasyn import MetaFrameBuilder
    from metasyn.registry import DistributionRegistry
    from collections import defaultdict

    return DistributionRegistry, MetaFrameBuilder, defaultdict, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set the size of the synthetic table. Warning: changing the number of columns will reset the specification of the columns.
    """)
    return


@app.cell
def _(mo):
    col_get_state, col_set_state = mo.state(3)
    row_get_state, row_set_state = mo.state(10)
    col_form = mo.ui.slider(1, 8, value=col_get_state(), show_value=True, label="Number of columns:  ", on_change=col_set_state)
    row_form = mo.ui.number(1, value=row_get_state(), label="Number of rows:  ", on_change=row_set_state)
    mo.hstack([col_form, row_form])
    #run_button = mo.ui.run_button(label="Submit")
    return col_form, row_form


@app.cell
def _(col_form, mo):
    get_name_state, set_name_state = mo.state([""]*col_form.value)
    def update_name_state(i, new_val):
        set_name_state(lambda cur: [
            new_val if idx == i else v for idx, v in enumerate(cur)
        ])

    get_var_state, set_var_state = mo.state([None]*col_form.value)
    def update_var_state(i, new_val):
        set_var_state(lambda cur: [
            new_val if idx == i else v for idx, v in enumerate(cur)
        ])

    get_unq_state, set_unq_state = mo.state([False]*col_form.value)
    def update_unq_state(i, new_val):
        set_unq_state(lambda cur: [
            new_val if idx == i else v for idx, v in enumerate(cur)
        ])

    get_dist_state, set_dist_state = mo.state([None]*col_form.value)
    def update_dist_state(i, new_val):
        set_dist_state(lambda cur: [
            new_val if idx == i else v for idx, v in enumerate(cur)
        ])

    get_param_state, set_param_state = mo.state([{} for _ in range(col_form.value)])
    def update_param_state(i, param_name, new_val, all_names):
        set_param_state(lambda cur: [
            {x: y for x, y in v.items() if x in all_names} | {param_name: new_val} if idx == i else v for idx, v in enumerate(cur)
        ])
    def update_param_defaults(i, default_params):
        cur_params = get_param_state()[i]
        if list(cur_params) == list(default_params) and [type(x) for x in cur_params.values()] == [type(x) for x in default_params.values()]:
            return
        set_param_state(lambda cur: [
            default_params if idx==i else v for idx, v in enumerate(cur)
        ])

    return (
        get_dist_state,
        get_name_state,
        get_param_state,
        get_unq_state,
        get_var_state,
        update_dist_state,
        update_name_state,
        update_param_defaults,
        update_param_state,
        update_unq_state,
        update_var_state,
    )


@app.cell
def _(col_form, get_name_state, mo, update_name_state):
    name_array = mo.ui.array(
        [
            mo.ui.text(value=get_name_state()[i], on_change=lambda v, i=i: update_name_state(i, v))
            for i in range(col_form.value)
        ]
    )
    return (name_array,)


@app.cell
def _(col_form, mo, update_var_state):
    var_array = mo.ui.array(
        [
            mo.ui.dropdown(options=["string", "categorical", "discrete", "date", "datetime", "time", "continuous"], on_change=lambda v, i=i: update_var_state(i, v))
            for i in range(col_form.value)
        ]
    )
    return (var_array,)


@app.cell
def _(col_form, mo, update_unq_state):
    unq_array = mo.ui.array(
        [
            mo.ui.checkbox(label=f"Unique?", on_change=lambda v, i=i: update_unq_state(i, v))
            for i in range(col_form.value)
        ]
    )
    return (unq_array,)


@app.cell
def _(DistributionRegistry, defaultdict):
    reg = DistributionRegistry.parse("builtin")

    def get_distributions(var_type, unique):
        try:
            distributions = reg.filter_distributions(var_type=var_type, unique=unique)
        except:
            return {}
        res_dict = defaultdict(list)
        for dist in distributions:
            if isinstance(dist.var_type, str):
                res_dict[dist.name].append(dist.var_type)
            else:
                res_dict[dist.name].extend(dist.var_type)
        return res_dict


    return get_distributions, reg


@app.cell
def _(
    col_form,
    get_dist_state,
    get_distributions,
    get_unq_state,
    get_var_state,
    mo,
    update_dist_state,
):
    dist_array = mo.ui.array(
        [
            mo.ui.dropdown(
                options=list(get_distributions(get_var_state()[i], get_unq_state()[i])),
                value=get_dist_state()[i] if get_dist_state()[i] in list(get_distributions(get_var_state()[i], get_unq_state()[i])) else None,
                on_change=lambda v, i=i: update_dist_state(i, v),
            )
            for i in range(col_form.value)
        ]
    )
    return (dist_array,)


@app.cell
def _(mo, np, reg, update_param_defaults, update_param_state):
    def get_parameters(dist_name, var_type, unique, idx):
        try:
            dist = reg.find_distribution(dist_name, var_type=var_type, unique=unique)
        except ValueError:
            return []
        params = dist.default_distribution().to_dict()["parameters"]
        if "regex_data" in params:
            params = {"regex_data": params["regex_data"]["regex"]}

        cur_param_form = []
        #cur_col_name = name_forms[m_col].value
        update_param_defaults(idx, params)
        for key, val in params.items():
            on_change = lambda v, idx=idx, param_name=key: update_param_state(idx, param_name, v, list(params))
            if isinstance(val, str) and dist.var_type == "datetime" and key in ["lower", "upper", "value"]:
                cur_param_form.append(mo.ui.datetime(value=val, label=key + ": ", on_change=on_change))
            elif isinstance(params[key], str):
                cur_param_form.append(mo.ui.text(value=val, label=key + ": " , on_change=on_change))
            elif isinstance(params[key], bool):
                cur_param_form.append(mo.ui.checkbox(value=val, label=key, on_change=on_change))
            elif isinstance(params[key], (float, int)):
                cur_param_form.append(mo.ui.number(value = val, label=key + ": ", on_change=on_change))
            elif isinstance(params[key], np.ndarray):
                if key == "probs":
                    array_change = lambda v, idx=idx, param_name=key: update_param_state(idx, param_name, [float(x.strip()) for x in v.split(",")], list(params))
                else:
                    array_change = lambda v, idx=idx, param_name=key: update_param_state(idx, param_name, v.split(","), list(params))

                cur_param_form.append(mo.ui.text(value=",".join(str(x) for x in val), label=key + ": ", on_change=array_change))
            
            else:
                raise ValueError("Not implemented type: ", type(params[key]), params[key])
        return cur_param_form

    return (get_parameters,)


@app.cell
def _(
    col_form,
    get_dist_state,
    get_parameters,
    get_unq_state,
    get_var_state,
    mo,
):
    param_array = mo.ui.array(
        [
            mo.ui.array(
                get_parameters(get_dist_state()[i], var_type=get_var_state()[i], unique=get_unq_state()[i], idx=i)
            )
            for i in range(col_form.value)
        ]
    )
    return (param_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create columns with distributions

    To generate synthetic data select the following for each column:

    - Name of the column
    - Type of the variable, e.g. string, discrete, time.
    - Uniqueness, whether the column has unique values.
    - Distribution, such as normal, exponential. The available distributions are only shown after selecting the variable type.
    - Parameters of the distribution.
    """)
    return


@app.cell
def _(dist_array, mo, name_array, param_array, unq_array, var_array):
    mo.vstack([
        mo.hstack(name_array),
        mo.hstack(var_array),
        mo.hstack(unq_array),
        mo.hstack(dist_array),
        mo.hstack(mo.vstack(x) for x in param_array)
    ])
    return


@app.cell
def _(
    MetaFrameBuilder,
    col_form,
    dist_array,
    get_param_state,
    name_array,
    row_form,
    unq_array,
    var_array,
):
    builder = MetaFrameBuilder()
    builder.n_rows = row_form.value
    for col_id in range(col_form.value):
        name = name_array[col_id].value
        name = f"Column {col_id}" if name == "" else name
        var_type = var_array[col_id].value
        dist_name = dist_array[col_id].value
        unique = unq_array[col_id].value
        parameters = get_param_state()[col_id]
        if var_type is None or dist_name is None or parameters is None:
            var_type = "string"
            dist_name = "na"
            parameters = {}
            unique = False
        builder.add_column(name=name, var_type=var_type)
        builder[name].distribution = {"name": dist_name, "unique": unique, "parameters": parameters}
    mf = builder.fit(progress_bar=False)
    mf.synthesize(progress_bar=False)
    return (mf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a GMF file

    Below you can generate a GMF (Generative Metadata Format) file. This file can be used to generate synthetic data. You can also generate synthetic data immediately one paragraph below.
    """)
    return


@app.cell
def _(mf, mo):
    from metasyn.metaframe import _jsonify
    import json

    def _to_json(mf):
        mf_dict = _jsonify(mf.to_dict())
        return json.dumps(mf_dict, indent=4).encode("utf-8")
    

    mo.download(
        data=_to_json(mf),
        filename="gmf.json",
        mimetype="application/json",
        label="Generate GMF file"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a CSV file

    With the button below you can download the CSV file with synthetic data.
    """)
    return


@app.cell
def _(mf, mo):
    from metasyn.file import CsvFileInterface
    from io import BytesIO

    def _gen_csv(mf):
        handler = BytesIO()
        mf.write_synthetic(handler, file_format=CsvFileInterface.default_interface("synthetic.csv").to_dict(), progress_bar=False)
        return handler

    mo.download(
        data=_gen_csv(mf),
        filename="synthetic.csv",
        mimetype="text/csv",
        label="Generate CSV",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
