import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean(x):
    x = re.sub(r"<[^>]+>|https?://\S+|www\.\S+|\S+@\S+", " ", str(x)[:10000])
    return re.sub(r"\s+", " ", x).strip()

def splits(path):
    df = pd.read_csv(path, compression="gzip")
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["row_id"] = df.index
    df["text"] = df.text.map(clean)
    df = df[df.text.str.len() >= 10].copy()
    df["key"] = df.text.str.casefold()
    df = df[df.groupby("key").sentiment.transform("nunique") == 1]
    # Preserve original validation as holdout; remove overlap from training.
    test = df[df.split == "validation"].drop_duplicates("key")
    pool = df[df.split == "train"].drop_duplicates("key")
    pool = pool[~pool.key.isin(test.key)]
    train, val = train_test_split(pool, test_size=.15, random_state=42, stratify=pool.sentiment)
    assert set(train.key).isdisjoint(val.key)
    assert set(train.key).isdisjoint(test.key)
    assert set(val.key).isdisjoint(test.key)
    return train, val, test

def export_splits(path, out):
    from pathlib import Path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("train","validation","test"),splits(path)):
        frame[["row_id","text","sentiment"]].to_csv(out / (name+".csv"),index=False)

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    export_splits(a.data,a.out)
