import pandas as pd


def load_data(path, label, *features):
    df = pd.read_csv(path)
    features, labels = df[list(*features)], df[label]
    return features, labels
