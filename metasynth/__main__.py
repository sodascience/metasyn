"""CLI for generating synthetic data frames from a metasynth .json file."""
import argparse
import json
import pathlib
import pickle
import sys

try:  # Python < 3.10 (backport)
    from importlib_metadata import entry_points
except ImportError:
    from importlib.metadata import entry_points  # type: ignore [assignment]

from metasynth import MetaFrame
from metasynth._version import version_tuple
from metasynth.validation import create_schema

SEMVER = f"{version_tuple[0]}.{version_tuple[1]}.{version_tuple[2]}"

MAIN_HELP_MESSAGE = f"""
Metasynth CLI version {SEMVER}

Usage: metasynth [subcommand] [options]

Available subcommands:
    synthesize - generate synthetic data from a .json file
    jsonschema - generate json schema from distribution providers

Program information:
    -v, --version - display CLI version and exit
    -h, --help    - display this help file and exit
"""

ENTRYPOINTS = ["synthesize", "jsonschema"]


def main() -> None:
    """CLI pointing to different entrypoints."""
    # show help by default, else consume first argument
    subcommand = "--help" if len(sys.argv) < 2 else sys.argv.pop(1)

    if subcommand in ["-h", "--help"]:
        print(MAIN_HELP_MESSAGE)
    elif subcommand in ["-v", "--version"]:
        print(f"Metasynth CLI version {SEMVER}")
    elif subcommand in ENTRYPOINTS:
        # find the subcommand in this module and run it!
        getattr(sys.modules[__name__], subcommand)()
    else:
        print(f"Invalid subcommand ({subcommand}). For help see metasynth --help")
        sys.exit(1)


def synthesize() -> None:
    """Program to generate synthetic data."""
    parser = argparse.ArgumentParser(
        prog="metasynth synthesize",
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
    if args.num_rows is not None:
        data_frame = meta_frame.synthesize(args.num_rows)
    else:
        data_frame = meta_frame.synthesize(meta_frame.n_rows)  # type: ignore [arg-type]

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


def jsonschema() -> None:
    """Program to generate json schema from dist providers."""
    parser = argparse.ArgumentParser(
        prog="metasynth jsonschema",
        description="Create Generative Metadata Format jsonschema and print to console.",
    )

    parser.add_argument(
        "plugins",
        help="Plugins to include in the generated schema (default none)",
        nargs="*",
    )

    parser.add_argument(
        "-a", "--avail",
        help="display available modules and quit",
        action="store_true",
    )

    # parse the args without the subcommand
    args, _ = parser.parse_known_args()

    # deduplicated list of plugins for schema
    plugins_avail = {entry.name for entry in entry_points(group="metasynth.distribution_provider")}

    if args.avail:
        _ = [print(a) for a in plugins_avail if a != "builtin"]
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
    schema = create_schema(list(plugins))
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
