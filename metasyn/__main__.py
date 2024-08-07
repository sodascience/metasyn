"""Module providing a Command Line Interface (CLI) for metasyn.

It provides functionality to generate GMF (.json) metadata files,
synthetic data from GMF files and creating json schemas for GMF files.
"""
import argparse
import json
import pathlib
import pickle
import sys
from argparse import RawDescriptionHelpFormatter

try:  # Python < 3.10 (backport)
    from importlib_metadata import entry_points, version
except ImportError:
    from importlib.metadata import entry_points, version  # type: ignore [assignment]

import polars as pl

from metasyn import MetaFrame
from metasyn.config import MetaConfig
from metasyn.validation import create_schema

EXAMPLE_CREATE_META="metasyn create-meta your_dataset.csv -o your_gmf_file.json --config your_config.toml"  # noqa # pylint: disable=line-too-long
EXAMPLE_SYNTHESIZE="metasyn synthesize your_gmf_file.json -o your_synthetic_file.csv"

MAIN_HELP_MESSAGE = f"""
Metasyn CLI version {version("metasyn")}

Usage: metasyn [subcommand] [options]

Available subcommands:
    create-meta:
        Create a intermediate metadata file (GMF/.json). This file can later be used to
        create a new synthetic dataset with the `synthesize` subcommand.
    synthesize:
        Create a synthetic dataset from the intermediate metadata file (GMF).
        To create a metadata file from your original dataset, use the `create-meta` subcommand.
    schema:
        Generate json schema from distribution providers.


To create a synthetic dataset from your original dataset you have to create a metadata file
and use this file to create a synthetic dataset.

WARNING: For the best results it is recommended to use the Python API. Things can and will go
wrong reading your dataset, and during the creation of metadata.

Example usage:

{EXAMPLE_CREATE_META}
{EXAMPLE_SYNTHESIZE}


Program information:
    -v, --version - display CLI version and exit
    -h, --help    - display this help file and exit
"""

ENTRYPOINTS = ["create-meta", "synthesize", "schema"]


def main() -> None:
    """CLI pointing to different entrypoints."""
    # show help by default, else consume first argument
    subcommand = "--help" if len(sys.argv) < 2 else sys.argv.pop(1)

    if subcommand in ["-h", "--help"]:
        print(MAIN_HELP_MESSAGE)
    elif subcommand in ["-v", "--version"]:
        print(f"Metasyn CLI version {version('metasyn')}")

    # find the subcommand in this module and run it!
    elif subcommand == "synthesize":
        synthesize()
    elif subcommand == "schema":
        schema()

    elif subcommand == "create-meta":
        create_metadata()
    else:
        print(f"Invalid subcommand ({subcommand}). For help see metasyn --help")
        sys.exit(1)


def create_metadata() -> None:
    """Program to create and export metadata from a DataFrame to a GMF file (.json)."""
    parser = argparse.ArgumentParser(
        prog="metasyn create-meta",
        description=f"""Create a Generative Metadata Format file from a CSV file.
This metadata file can then be used to create a synthetic dataset with the `synthesize` subcommand.

Example: {EXAMPLE_CREATE_META}
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="input file; a CSV file that you want to synthesize later.",
        type=pathlib.Path,
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--output", "-o",
        help="Metadata GMF output file: .json. This file can be used to synthesize data.",
        type=pathlib.Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--config",
        help="Configuration file (*.toml) to improve the quality and/or privacy of the metadata.",
        type=pathlib.Path,
        default=None,
    )

    args, _ = parser.parse_known_args()
    if args.config is not None:
        meta_config = MetaConfig.from_toml(args.config)
    else:
        meta_config = None

    if args.input is None:
        if meta_config is None:
            raise ValueError("Please supply either an input dataset or a configuration file.")
        meta_frame = MetaFrame.from_config(meta_config)
    else:
        data_frame = pl.read_csv(args.input, try_parse_dates=True, infer_schema_length=10000,
                                 null_values=["", "na", "NA", "N/A", "Na"],
                                 ignore_errors=True)
        meta_frame = MetaFrame.fit_dataframe(data_frame, meta_config)
    meta_frame.export(args.output)


def synthesize() -> None:
    """Program to generate synthetic data."""
    parser = argparse.ArgumentParser(
        prog="metasyn synthesize",
        description=f"""Synthesize data from a Generative Metadata Format .json file.
To create the metadata file from your dataset, use the `create-meta` subcommand.

Example: {EXAMPLE_SYNTHESIZE}
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="input file; typically .json adhering to the Generative Metadata Format",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output", "-o",
        help="output file (.csv, .feather, .parquet, .pkl, or .xlsx)",
        type=pathlib.Path,
        required=False,
    )
    parser.add_argument(
        "-n", "--num_rows",
        help="number of rows to synthesize",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-p", "--preview",
        help="preview six-row synthesized data frame in console and exit",
        action="store_true",
    )

    # parse the args without the subcommand
    args, _ = parser.parse_known_args()

    if not args.preview and not args.output:
        parser.error("Output file is required if you are not using the preview option.")

    # Create the metaframe from the json file
    try:
        meta_frame = MetaFrame.from_json(args.input)
    except json.JSONDecodeError as _err:
        print(f"Error: Unable to parse the file '{args.input}'.\n\n"
              "Expecting a GMF/.json file as input.\n"
              "Did you perhaps provide your dataset?\n"
              "If so, please first create the metadata with the `create-meta` sub command.\n"
              "Otherwise your GMF file might be corrupted, and you should recreate it.")
        return

    if args.preview:
        # only print six rows and exit
        print(meta_frame.synthesize(6))
        return

    # Generate a data frame
    data_frame = meta_frame.synthesize(args.num_rows)

    # Store the dataframe to file
    if args.output.suffix == ".csv":
        data_frame.write_csv(args.output)
    elif args.output.suffix == ".feather":
        data_frame.write_ipc(args.output)
    elif args.output.suffix == ".parquet":
        data_frame.write_parquet(args.output)
    elif args.output.suffix == ".xlsx":
        data_frame.write_excel(args.output)
    elif args.output.suffix == ".pkl":
        with args.output.open("wb") as pkl_file:
            pickle.dump(data_frame, file=pkl_file)
    else:
        parser.error(
            f"Unsupported output file format ({args.output.suffix})."
            "Use .csv, .feather, .parquet, .pkl, or .xlsx.",
        )


def schema() -> None:
    """Program to generate json schema from dist providers."""
    parser = argparse.ArgumentParser(
        prog="metasyn schema",
        description="Create Generative Metadata Format schema and print to console.",
    )

    parser.add_argument(
        "plugins",
        help="Plugins to include in the generated schema (default builtin)",
        nargs="*"
    )

    parser.add_argument(
        "-l", "--list",
        help="display available plugins and quit",
        action="store_true",
    )

    # parse the args without the subcommand
    args, _ = parser.parse_known_args()

    # deduplicated list of plugins for schema
    plugins_avail = {entry.name for entry in entry_points(group="metasyn.distribution_provider")}

    if args.list:
        for a in plugins_avail:
            print(a)
        return

    plugins = {"builtin", *args.plugins}
    if len(plugins - plugins_avail) > 0:
        notfound = ", ".join(plugins - plugins_avail)
        pl_avail = ", ".join(plugins_avail - {"builtin"})
        errmsg = (
            f"\n  Requested plugin(s) not found: {notfound}"
            f"\n  Available plugins: {pl_avail}"
        )
        parser.error(errmsg)
    jsonschema = create_schema(list(plugins))
    print(json.dumps(jsonschema, indent=2))


if __name__ == "__main__":
    main()
