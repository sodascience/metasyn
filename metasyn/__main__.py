"""Module providing a Command Line Interface (CLI) for metasyn.

It provides functionality to generate GMF (.json) metadata files,
synthetic data from GMF files and creating json schemas for GMF files.
"""
import argparse
import json
import pathlib
from typing import Optional
import sys
from argparse import RawDescriptionHelpFormatter

from importlib.metadata import entry_points, version

from metasyn import MetaFrame
from metasyn.config import MetaConfig
from metasyn.file import file_interface_from_dict, get_file_interface_class, read_file
from metasyn.validation import create_schema

EXAMPLE_CREATE_META="metasyn create-meta your_dataset.csv -o your_gmf_file.json --config your_config.toml" # noqa: E501
EXAMPLE_CREATE_TOML="metasyn create-meta your_dataset.csv -o your_gmf_file.toml --config your_config.toml" # noqa: E501
EXAMPLE_SYNTHESIZE="metasyn synthesize your_gmf_file.json -o your_synthetic_file.csv"

MAIN_HELP_MESSAGE = f"""
Metasyn CLI version {version("metasyn")}

Usage: metasyn [subcommand] [options]

Available subcommands:
    create-meta:
        Create a intermediate metadata file (GMF/.json/.toml). This file can later be used to
        create a new synthetic dataset with the `synthesize` subcommand.
    synthesize:
        Create a synthetic dataset from the intermediate metadata file (GMF).
        To create a metadata file from your original dataset, use the `create-meta` subcommand.
    schema:
        Generate json schema from distribution registries.


To create a synthetic dataset from your original dataset you have to create a metadata file
and use this file to create a synthetic dataset.

WARNING: For the best results it is recommended to use the Python API. Things can and will go
wrong reading your dataset, and during the creation of metadata.

Example usage:

{EXAMPLE_CREATE_META}
{EXAMPLE_CREATE_TOML}
{EXAMPLE_SYNTHESIZE}


Program information:
    -v, --version - display CLI version and exit
    -h, --help    - display this help file and exit
"""

ENTRYPOINTS = ["create-meta", "synthesize", "schema"]


def main(input_args: Optional[list[str]] = None) -> None:
    """CLI pointing to different entrypoints."""
    # show help by default, else consume first argument
    if input_args is None:
        input_args = sys.argv[1:]
    else:
        input_args = [str(x) for x in input_args]
    if len(input_args) == 0:
        input_args = ["--help"]

    subcommand = input_args[0]
    input_args.pop(0)
    if subcommand in ["-h", "--help"]:
        print(MAIN_HELP_MESSAGE)
    elif subcommand in ["-v", "--version"]:
        print(f"Metasyn CLI version {version('metasyn')}")

    # find the subcommand in this module and run it!
    elif subcommand == "synthesize":
        synthesize(input_args)
    elif subcommand == "schema":
        schema(input_args)
    elif subcommand == "create-meta":
        create_metadata(input_args)
    else:
        print(f"Invalid subcommand ({subcommand}). For help see metasyn --help")
        sys.exit(1)


def create_metadata(input_args) -> None:
    """Program to create and save metadata from a DataFrame to a GMF file (.json/.toml)."""
    parser = argparse.ArgumentParser(
        prog="metasyn create-meta",
        description=f"""Create a Generative Metadata Format file from a CSV file.
This metadata file can then be used to create a synthetic dataset with the `synthesize` subcommand.

Examples:

{EXAMPLE_CREATE_META}
{EXAMPLE_CREATE_TOML}
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
        help="Metadata GMF output file: .json/.toml. This file can be used to synthesize data.",
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

    args, _ = parser.parse_known_args(input_args)
    if args.config is not None:
        meta_config = MetaConfig.from_toml(args.config)
    else:
        meta_config = None

    if args.input is None:
        if meta_config is None:
            raise parser.error("Please supply either an input dataset or a configuration file.")
        meta_frame = MetaFrame.from_config(meta_config)
    else:
        if meta_config is not None and meta_config.file_config is not None:
            data_frame, file_handler = read_file(args.input, **meta_config.file_config)
        else:
            data_frame, file_handler = read_file(args.input)
        meta_frame = MetaFrame.fit_dataframe(data_frame, config=meta_config)
        meta_frame.file_format = file_handler.to_dict()
    meta_frame.save(args.output)


def synthesize(input_args) -> None:
    """Program to generate synthetic data."""
    parser = argparse.ArgumentParser(
        prog="metasyn synthesize",
        description=f"""Synthesize data from a Generative Metadata Format .json/.toml file.
To create the metadata file from your dataset, use the `create-meta` subcommand.

Example: {EXAMPLE_SYNTHESIZE}
""",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="input file; typically .json or .toml adhering to the Generative Metadata Format",
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
        "-s", "--seed",
        help="Seed for the generation of synthetic data.",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-p", "--preview",
        help="preview six-row synthesized data frame in console and exit",
        action="store_true",
    )

    # parse the args without the subcommand
    args, _ = parser.parse_known_args(input_args)

    if not args.preview and not args.output:
        parser.error("Output file is required if you are not using the preview option.")

    # Create the metaframe from the GMF file
    try:
        meta_frame = MetaFrame.load(args.input)
    except (json.JSONDecodeError, UnicodeDecodeError) as _err:
        parser.error(
            f"Unable to parse the file '{args.input}'.\n\n"
            "Expecting a GMF/.json/.toml file as input.\n"
            "Did you perhaps provide your dataset?\n"
            "If so, please first create the metadata with the `create-meta` sub command.\n"
            "Otherwise your GMF file might be corrupted, and you should recreate it.")
        return

    if args.preview:
        # only print six rows and exit
        print(meta_frame.synthesize(6, seed=args.seed))
        return


    # Store the dataframe to file
    if meta_frame.file_format is not None:
        file_interface = file_interface_from_dict(meta_frame.file_format)
        if args.output.suffix not in file_interface.extensions:
            file_interface = get_file_interface_class(args.output).default_interface(args.output)
        meta_frame.write_synthetic(args.output, n=args.num_rows, seed=args.seed,
                                   file_format=file_interface)
    else:
        file_interface = get_file_interface_class(args.output).default_interface(args.output)
        meta_frame.write_synthetic(args.output, n=args.num_rows, seed=args.seed,
                                   file_format=file_interface)


def schema(input_args) -> None:
    """Program to generate json schema from dist registries."""
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

    parser.add_argument(
        "-o", "--output",
        help="File to write the schema to.",
        type=pathlib.Path,
    )

    # parse the args without the subcommand
    args = parser.parse_args(input_args)

    # deduplicated list of plugins for schema
    plugins_avail = {entry.name for entry in entry_points(group="metasyn.distribution_registry")}

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
    if args.output is None:
        print(json.dumps(jsonschema, indent=4))
    else:
        with open(args.output, "w") as handle:
            json.dump(jsonschema, handle, indent=4)


if __name__ == "__main__":
    main()
