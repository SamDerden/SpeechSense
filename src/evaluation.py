import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, roc_auc_score, log_loss)
def evaluate(y, probabilities):
    y=np.asarray(y)
    p=np.asarray(probabilities)
    if p.shape != (len(y),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError("Invalid three-class probabilities")
    pred=p.argmax(1)
    one=np.eye(3)[y.astype(int)]
    result={"accuracy":accuracy_score(y,pred),"log_loss":log_loss(y,p,labels=[0,1,2])}
    for avg in ("macro","weighted","micro"):
        for name,fn in (("f1",f1_score),("precision",precision_score),("recall",recall_score)):
            result[avg+"_"+name]=fn(y,pred,labels=[0,1,2],average=avg,zero_division=0)
    result["macro_pr_auc_ap"]=float(average_precision_score(one,p,average="macro")) if len(set(y))==3 else None
    result["macro_roc_auc"]=float(roc_auc_score(y,p,multi_class="ovr",labels=[0,1,2])) if len(set(y))==3 else None
    result["per_class"]=classification_report(y,pred,labels=[0,1,2],target_names=["neutral","positive","negative"],output_dict=True,zero_division=0)
    result["confusion_matrix"]=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    return result
