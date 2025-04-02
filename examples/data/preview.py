#!/usr/bin/env python

from metasyn.filereader import get_file_reader
import sys


if __name__ == "__main__":
    dataset_fp = sys.argv[1]
    n_rows=6
    # n_rows = int(sys.argv[2])
    df, handler = get_file_reader(dataset_fp)
    # df, _ = handler_class.from_file(dataset_fp)
    print(df.head(n_rows))