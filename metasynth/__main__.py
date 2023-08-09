"""CLI for generating synthetic data frames from a metasynth .json file."""
import argparse
import pathlib
import pickle
import sys

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
        "-p", "--print_only",
        help="print six-row data frame to console and exit",
        action="store_true"
    )
    # version
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args, _ = parser.parse_known_args()

    if not args.print_only and not args.output:
        parser.error("Output file is required.")

    # Create the metaframe from the json file
    meta_frame = MetaFrame.from_json(args.input)

    if args.print_only:
        # only print six rows and exit
        print(meta_frame.synthesize(6))
        sys.exit(0)

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
            pkl_file.close()
    else:
        parser.error(
            f"Output file format ({args.output.suffix}) incorrect." +
            "Use .csv, .feather, .parquet, .pkl, or .xlsx."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
