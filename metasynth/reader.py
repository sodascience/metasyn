import pandas as pd


class CSVReader():
    def __init__(self, fp):
        self.df = pd.read_csv(fp)
        print(self.df)
