import pandas as pd


# Load CSV dataset
def load_dataset(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    return df
