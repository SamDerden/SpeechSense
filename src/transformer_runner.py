"""Run one experiment per process. Python 3.10+, transformers 4.46–4.49."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback)
from transformers.trainer_utils import get_last_checkpoint

MODELS = {
    "tiny": "cointegrated/rubert-tiny2",
    "rubert": "DeepPavlov/rubert-base-cased",
    "sentiment": "blanchefort/rubert-base-cased-sentiment-rusentiment",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "large": "sberbank-ai/ruRoberta-large",
}

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

def metrics(p):
    pred = np.argmax(p.predictions, axis=-1)
    return {"macro_f1": f1_score(p.label_ids, pred, average="macro", labels=[0,1,2], zero_division=0),
            "accuracy": accuracy_score(p.label_ids, pred)}

class WeightedTrainer(Trainer):
    def __init__(self, *args, weights=None, differential=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.differential = differential
        self.technique = "baseline"
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        if self.technique == "mixup" and model.training:
            emb = model.get_input_embeddings()(inputs["input_ids"])
            perm = torch.randperm(len(labels), device=labels.device)
            lam = float(np.random.beta(.2,.2))
            mask = inputs["attention_mask"]
            out = model(inputs_embeds=lam*emb+(1-lam)*emb[perm], attention_mask=torch.maximum(mask,mask[perm]))
            loss = lam*torch.nn.functional.cross_entropy(out.logits,labels)+(1-lam)*torch.nn.functional.cross_entropy(out.logits,labels[perm])
            return (loss,out) if return_outputs else loss
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(out.logits, labels,
            weight=self.weights.to(out.logits.device) if self.weights is not None else None,
            label_smoothing=self.args.label_smoothing_factor)
        return (loss, out) if return_outputs else loss
    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.technique != "fgm":
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, dict(inputs))
        emb = self.accelerator.unwrap_model(model).get_input_embeddings().weight
        grad = torch.autograd.grad(loss, emb, retain_graph=True)[0]
        self.accelerator.backward(loss / self.args.gradient_accumulation_steps)
        delta = grad / grad.norm().clamp_min(1e-12)
        with torch.no_grad():
            emb.add_(delta)
        try:
            with self.compute_loss_context_manager():
                adv = self.compute_loss(model, dict(inputs))
            self.accelerator.backward(adv / self.args.gradient_accumulation_steps)
        finally:
            with torch.no_grad():
                emb.sub_(delta)
        return (loss.detach()+adv.detach()) / self.args.gradient_accumulation_steps
    def create_optimizer(self):
        if not self.differential:
            return super().create_optimizer()
        if self.optimizer is None:
            groups = {}
            base = {id(p) for p in self.model.base_model.parameters()}
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                lr = self.args.learning_rate * (.2 if id(p) in base else 5)
                decay = 0.0 if name.endswith("bias") or p.ndim == 1 else self.args.weight_decay
                groups.setdefault((lr, decay), []).append(p)
            self.optimizer = torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": d}
                for (lr,d),ps in groups.items()])
        return self.optimizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--technique", choices=["baseline","weighted","differential","smoothing","clipping","custom_head","mean_pool","mixup","fgm","back_translation","contextual"], default="baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--test", action="store_true", help="Only after model selection is frozen")
    args = p.parse_args()
    from transformers import set_seed
    set_seed(args.seed)
    train,val,test = splits(args.data)
    if args.smoke:
        train,val = train.sample(min(96,len(train)),random_state=42),val.sample(min(48,len(val)),random_state=42)
        if args.test: raise ValueError("Smoke run cannot evaluate final test")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("test")
    manifest = out / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Use a new output directory for a different configuration")
    manifest.write_text(json.dumps(config,indent=2))
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model],num_labels=3,
        ignore_mismatched_sizes=True, id2label={0:"neutral",1:"positive",2:"negative"},
        label2id={"neutral":0,"positive":1,"negative":2})
    if args.technique in ("custom_head","mean_pool"):
        from transformer_heads import install_head
        install_head(model, args.technique)
    # Existing sentiment heads may use a different class order; reset classifier.
    if args.model == "sentiment" and args.technique not in ("custom_head","mean_pool"):
        model.classifier.apply(model._init_weights)
    def encode(frame):
        ds = Dataset.from_dict({"text":frame.text.tolist(),"labels":frame.sentiment.astype(int).tolist()})
        return ds.map(lambda b:tok(b["text"],truncation=True,max_length=128),batched=True,remove_columns=["text"])
    if args.technique in ("back_translation","contextual"):
        from transformer_augmentation import augment
        cache = out / "augmentation.csv"
        if cache.exists():
            extra = pd.read_csv(cache)
        else:
            extra = augment(train,args.technique,limit=4 if args.smoke else 3000,seed=args.seed,device=0 if torch.cuda.is_available() else -1)
            extra.to_csv(cache,index=False)
        forbidden = set(val.text.str.casefold()) | set(test.text.str.casefold())
        extra = extra[~extra.text.str.casefold().isin(forbidden)]
        train = pd.concat([train,extra],ignore_index=True)
    tr,va = encode(train),encode(val)
    weights = None
    if args.technique == "weighted":
        counts = train.sentiment.value_counts().reindex([0,1,2],fill_value=0).values
        if min(counts)==0: raise ValueError("Missing class")
        weights = torch.tensor(len(train)/(3*counts),dtype=torch.float32)
    ta = TrainingArguments(output_dir=str(out),eval_strategy="epoch",save_strategy="epoch",
        load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,
        save_total_limit=2,num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch,per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,learning_rate=2e-5,weight_decay=.01,
        lr_scheduler_type="cosine",warmup_ratio=.1,fp16=torch.cuda.is_available(),
        label_smoothing_factor=.1 if args.technique=="smoothing" else 0.,
        max_grad_norm=.5 if args.technique=="clipping" else 1.,
        seed=args.seed,report_to=[],logging_steps=50)
    trainer = WeightedTrainer(model=model,args=ta,train_dataset=tr,eval_dataset=va,
        data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics,
        weights=weights,differential=args.technique=="differential",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.technique = args.technique
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)))
    trainer.save_model(str(out/"best"))
    tok.save_pretrained(out/"best")
    for name,frame,ds in [("validation",val,va)]+([("test",test,encode(test))] if args.test else []):
        result=trainer.predict(ds)
        probs=torch.softmax(torch.tensor(result.predictions),dim=-1).numpy()
        pd.DataFrame({"row_id":frame.row_id.values,"label":frame.sentiment.values,
            "p0":probs[:,0],"p1":probs[:,1],"p2":probs[:,2]}).to_csv(out/f"{name}_predictions.csv",index=False)
        (out/f"{name}_metrics.json").write_text(json.dumps(result.metrics,indent=2))
    (out/"history.json").write_text(json.dumps(trainer.state.log_history,indent=2))

if __name__ == "__main__":
    main()
