import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def load_predictions(paths):
    frames = [pd.read_csv(p).sort_values("row_id").reset_index(drop=True) for p in paths]
    if not frames:
        raise ValueError("No predictions")
    ref = frames[0]
    for f in frames:
        if f.row_id.duplicated().any() or not f[["row_id","label"]].equals(ref[["row_id","label"]]):
            raise ValueError("Prediction rows or labels differ")
        p = f[["p0","p1","p2"]].to_numpy()
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError("Invalid probabilities")
    return ref, np.stack([f[["p0","p1","p2"]].to_numpy() for f in frames])

def predict(p, thresholds):
    adjusted = np.where(p >= thresholds,p,0)
    return np.where(adjusted.sum(1)>0,adjusted.argmax(1),p.argmax(1))

def tune(p,y):
    from itertools import product
    grid = product(np.arange(.25,.71,.05),repeat=3)
    return max(grid,key=lambda t:f1_score(y,predict(p,t),labels=[0,1,2],average="macro",zero_division=0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--validation",nargs="+",required=True)
    ap.add_argument("--test",nargs="+")
    ap.add_argument("--out",required=True)
    ap.add_argument("--thresholds",action="store_true")
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ref,stack=load_predictions(a.validation)
    p=stack.mean(0)
    thresholds=list(tune(p,ref.label)) if a.thresholds else [0.,0.,0.]
    (out/"thresholds.json").write_text(json.dumps(thresholds))
    def save(name,ref,p):
        pred=predict(p,thresholds)
        f=ref[["row_id","label"]].copy()
        f[["p0","p1","p2"]]=p
        f["prediction"]=pred
        f.to_csv(out/f"{name}_predictions.csv",index=False)
        (out/f"{name}_metrics.json").write_text(json.dumps({"macro_f1":f1_score(ref.label,pred,labels=[0,1,2],average="macro",zero_division=0),"accuracy":accuracy_score(ref.label,pred)}))
    save("validation",ref,p)
    if a.test:
        if len(a.test)!=len(a.validation):raise ValueError("Model counts differ")
        ref,stack=load_predictions(a.test)
        save("test",ref,stack.mean(0))

if __name__=="__main__":
    main()
