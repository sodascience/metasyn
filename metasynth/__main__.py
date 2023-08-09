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
        help="Input file.",
        type=pathlib.Path
    )
    parser.add_argument(
        "output",
        help="Output file. File extension should be one of: .csv, .feather, .parquet, .pkl, .xlsx",
        type=pathlib.Path
    )
    parser.add_argument(
        "-n", 
        "--num_rows",
        help="Number of rows to synthesize",
        default=None,
        type=int,
        required=False
    )
    parser.add_argument(
        "-p", "--print_only",
        help="print one-row data frame to console and exit",
        action="store_true"
    )
    # version
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args, _ = parser.parse_known_args()

    print(args)
    # Create the metaframe from the json file
    mf = MetaFrame.from_json(args.input)

    if args.print_only:
        # only print one row and exit
        print(mf.synthesize(1))
        sys.exit(0)

    # Generate a data frame
    if args.num_rows is not None:
        df = mf.synthesize(args.num_rows)
    else:
        df = mf.synthesize(mf.n_rows)

    # Store the dataframe to file
    match args.output.suffix:
        case ".csv":
            df.write_csv(args.output)
        case ".feather":
            df.write_ipc(args.output)
        case ".parquet":
            df.write_parquet(args.output)
        case ".xlsx":
            df.write_excel(args.output)
        case ".pkl":
            with open(args.output, "wb") as f:
                pickle.dump(df, file=f)
                f.close()
        case _:
            print(
                f"Output file format ({args.output.suffix}) incorrect.",
                "Use .csv, .feather, .parquet, .pkl, or .xlsx."
            )
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
