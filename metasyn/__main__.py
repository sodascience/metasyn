"""CLI for generating synthetic data frames from a metasyn .json file."""
import argparse
import json
import pathlib
import pickle
import sys
from configparser import ConfigParser

try:  # Python < 3.10 (backport)
    from importlib_metadata import entry_points, version
except ImportError:
    from importlib.metadata import entry_points, version  # type: ignore [assignment]

import polars as pl

from metasyn import MetaFrame
from metasyn.validation import create_schema

MAIN_HELP_MESSAGE = f"""
Metasyn CLI version {version("metasyn")}

Usage: metasyn [subcommand] [options]

Available subcommands:
    create-meta - generate a GMF (.json) file.
    synthesize - generate synthetic data from a GMF (.json) file
    jsonschema - generate json schema from distribution providers

Program information:
    -v, --version - display CLI version and exit
    -h, --help    - display this help file and exit
"""

ENTRYPOINTS = ["synthesize", "schema"]


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


def _parse_config(config_fp):
    config = ConfigParser()
    config.read(config_fp)
    spec = {}
    for section in config.sections():
        if section.startswith("var."):
            new_dict = {}
            for key, val in dict(config[section]).items():
                try:
                    new_dict[key] = config.getboolean(section, key)
                except ValueError:
                    pass
                try:
                    new_dict[key] = config.getfloat(section, key)
                except ValueError:
                    pass
                try:
                    new_dict[key] = config.getint(section, key)
                except ValueError:
                    pass
                if key not in new_dict:
                    new_dict[key] = val
            spec[section[4:]] = new_dict
    return spec


def create_metadata():
    """Program to create and export metadata from a DataFrame to a GMF file (.json)."""
    parser = argparse.ArgumentParser(
        prog="metasyn create-meta",
        description="Create a Generative Metadata Format file from a CSV file.",
    )
    parser.add_argument(
        "input",
        help="input file; a CSV file that you want to synthesize later.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="output file: .json",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--config",
        help="Configuration file to specify distribution behavior.",
        type=pathlib.Path,
        default=None,
    )
    args, _ = parser.parse_known_args()
    if args.config is not None:
        spec = _parse_config(args.config)
    else:
        spec = {}
    data_frame = pl.read_csv(args.input, try_parse_dates=True)
    meta_frame = MetaFrame.fit_dataframe(data_frame, spec=spec)
    meta_frame.export(args.output)


def synthesize() -> None:
    """Program to generate synthetic data."""
    parser = argparse.ArgumentParser(
        prog="metasyn synthesize",
        description="Synthesize data from Generative Metadata Format .json file.",
    )
    parser.add_argument(
        "input",
        help="input file; typically .json adhering to the Generative Metadata Format",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="output file (.csv, .feather, .parquet, .pkl, or .xlsx)",
        nargs="?",
        type=pathlib.Path,
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
        parser.error("Output file is required.")

    # Create the metaframe from the json file
    meta_frame = MetaFrame.from_json(args.input)

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
