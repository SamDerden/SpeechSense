import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
from transformer_ensemble import load_predictions, tune, predict
from evaluation import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    root=Path(a.out).resolve()
    root.mkdir(parents=True,exist_ok=True)
    runner=Path(__file__).with_name("transformer_runner.py")
    def run(model,tech="baseline",seed=42,extra=()):
        dest=root/f"{model}_{tech}_{seed}"
        cmd=[sys.executable,str(runner),"--data",a.data,"--out",str(dest),"--model",model,"--technique",tech,"--seed",str(seed),*(["--smoke"] if a.smoke else []),*extra]
        subprocess.run(cmd,check=True)
        return dest
    def score(d):
        return json.loads((d/"validation_metrics.json").read_text())["macro_f1"]
    def ensemble(name,dirs,threshold=False):
        ref,stack=load_predictions([d/"validation_predictions.csv" for d in dirs])
        probs=stack.mean(0)
        t=list(tune(probs,ref.label)) if threshold else [0.,0.,0.]
        pred=predict(probs,t)
        out=root/name
        out.mkdir(exist_ok=True)
        report=evaluate(ref.label.to_numpy(),probs)
        report["decision_macro_f1"]=f1_score(ref.label,pred,labels=[0,1,2],average="macro",zero_division=0)
        report["members"]=[str(d) for d in dirs]
        report["thresholds"]=t
        (out/"validation_metrics.json").write_text(json.dumps(report,indent=2))
        ref=ref[["row_id","label"]].copy()
        ref[["p0","p1","p2"]]=probs
        ref["prediction"]=pred
        ref.to_csv(out/"validation_predictions.csv",index=False)
    models=["tiny","rubert","sentiment"] if a.smoke else ["tiny","rubert","sentiment","xlmr","large"]
    baselines={m:run(m) for m in models}
    ranked=sorted(models,key=lambda m:score(baselines[m]),reverse=True)
    best=ranked[0]
    teacher=run(best,extra=["--export-teacher"])
    run("tiny","distillation",extra=["--teacher-logits",str(teacher/"teacher_logits.csv")])
    ensemble("top3",[baselines[m] for m in ranked[:3]])
    ensemble("thresholds",[baselines[best]],True)
    ensemble("seed_ensemble",[baselines[best],run(best,seed=43),run(best,seed=44)])
    for tech in ["weighted","differential","smoothing","clipping","custom_head","mean_pool","mixup","fgm","curriculum","multitask","active","back_translation","contextual"]:
        run(best,tech)
    (root/"suite_complete.json").write_text(json.dumps({"best":best,"ranked":ranked,"smoke":a.smoke},indent=2))
if __name__=="__main__":
    main()
