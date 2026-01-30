import pandas as pd
from pathlib import Path

# Fichiers à modifier
files = [
    '../data/processed/v1/test.csv',
    '../data/processed/v2/test.csv',
    '../data/processed/v3/test.csv',
]

     # Injecte un drift sur la colonne num_users (4e colonne, index 3)

def inject_drift_v1(file_path):
    # num_users (col 3), 100 lignes, +5
    path = Path(__file__).parent / file_path
    df = pd.read_csv(path)
    colname = df.columns[3]
    df.loc[:99, colname] += 5.0
    df.to_csv(path, index=False)
    print(f"Drift injecté dans {file_path} sur {colname} (delta=+5, n=100)")

def inject_drift_v2(file_path):
    # avg_age (col 4), 50 lignes, +3
    path = Path(__file__).parent / file_path
    df = pd.read_csv(path)
    colname = df.columns[4]
    df.loc[:49, colname] += 3.0
    df.to_csv(path, index=False)
    print(f"Drift injecté dans {file_path} sur {colname} (delta=+3, n=50)")

def inject_drift_v3(file_path):
    # Drift fort sur plusieurs colonnes pour déclencher une alerte critique
    path = Path(__file__).parent / file_path
    df = pd.read_csv(path)
    # min_age (col 6), max_age (col 7), avg_age (col 4), num_users (col 3)
    cols = [3, 4, 6, 7]
    deltas = [10, 8, 6, -12]  # valeurs fortes pour garantir le drift
    n = 60  # nombre de lignes modifiées
    for col, delta in zip(cols, deltas):
        colname = df.columns[col]
        df.loc[:n-1, colname] += delta
        print(f"Drift injecté dans {file_path} sur {colname} (delta={delta}, n={n})")
    df.to_csv(path, index=False)

if __name__ == "__main__":
    inject_drift_v1('../data/processed/v1/test.csv')
    inject_drift_v2('../data/processed/v2/test.csv')
    inject_drift_v3('../data/processed/v3/test.csv')
