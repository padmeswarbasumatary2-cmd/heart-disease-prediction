import pandas as pd

def load_data(path: str):
    """Load CSV dataset from path and return DataFrame."""
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/heart.csv'
    df = load_data(path)
    print('Loaded', len(df), 'rows from', path)