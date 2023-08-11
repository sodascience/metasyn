"""CLI for generating synthetic data frames from a metasynth .json file."""
import argparse
import pathlib
import pickle

from metasynth._version import __version__
from metasynth import MetaFrame


def main():
    """Parse arguments and generate synthetic data."""
    parser = argparse.ArgumentParser(
        prog="metasynth",
        description="Synthesize data from Generative Metadata Format .json file.",
    )
    parser.add_argument(
        "input",
        help="input file; typically .json adhering to the Generative Metadata Format",
        type=pathlib.Path
    )
    parser.add_argument(
        "output",
        help="output file (.csv, .feather, .parquet, .pkl, or .xlsx)",
        nargs="?",
        type=pathlib.Path
    )
    parser.add_argument(
        "-n", "--num_rows",
        help="number of rows to synthesize",
        default=None,
        type=int,
        required=False
    )
    parser.add_argument(
        "-p", "--preview",
        help="preview six-row synthesized data frame in console and exit",
        action="store_true"
    )
    # version
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

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
        data_frame = meta_frame.synthesize(meta_frame.n_rows)

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
        with open(args.output, "wb") as pkl_file:
            pickle.dump(data_frame, file=pkl_file)
    else:
        parser.error(
            f"Unsupported output file format ({args.output.suffix})." +
            "Use .csv, .feather, .parquet, .pkl, or .xlsx."
        )


if __name__ == "__main__":
    main()
