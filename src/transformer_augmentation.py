import pandas as pd
import random

def augment(train, technique, limit=3000, seed=42, device=-1):
    rng=random.Random(seed)
    texts=train.loc[train.sentiment==0,"text"].drop_duplicates().tolist()
    rng.shuffle(texts)
    if technique=="back_translation":
        from transformers import pipeline
        forward=pipeline("translation",model="Helsinki-NLP/opus-mt-ru-en",device=device)
        backward=pipeline("translation",model="Helsinki-NLP/opus-mt-en-ru",device=device)
        def transform(t):
            return backward(forward(t,truncation=True,max_length=256)[0]["translation_text"],truncation=True,max_length=256)[0]["translation_text"]
    elif technique=="contextual":
        import nlpaug.augmenter.word as naw
        aug=naw.ContextualWordEmbsAug(model_path="DeepPavlov/rubert-base-cased",action="substitute",aug_min=1,aug_max=3,aug_p=.2,device="cuda" if device>=0 else "cpu")
        def transform(t):
            result=aug.augment(t)
            return result[0] if isinstance(result,list) and result else result
    else:
        raise ValueError(technique)
    seen=set(train.text.str.casefold())
    rows=[]
    for t in texts[:limit]:
        value=transform(t)
        if isinstance(value,str) and len(value.strip())>=10 and value.strip().casefold() not in seen:
            value=value.strip()
            seen.add(value.casefold())
            rows.append({"text":value,"sentiment":0})
    if not rows:
        raise RuntimeError("No unique augmented texts")
    return pd.DataFrame(rows)
