import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

def uncertainty_indices(logits, count):
    p=torch.as_tensor(logits).float().softmax(-1)
    entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
    return torch.argsort(entropy,descending=True,stable=True)[:count].cpu().numpy()

class CurriculumSampler(Sampler):
    """Easy-to-hard order, full coverage per epoch."""
    def __init__(self,lengths,seed=42):
        self.lengths=np.asarray(lengths)
        self.seed=seed
        self.epoch=0
    def __len__(self):
        return len(self.lengths)
    def set_epoch(self,epoch):
        self.epoch=epoch
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        ordered=np.argsort(self.lengths,kind="stable")
        groups=np.array_split(ordered,5)
        for g in groups:
            rng.shuffle(g)
        return iter(np.concatenate(groups).tolist())

def distillation_loss(logits,labels,teacher_logits,temperature=2.,alpha=.5):
    ce=nn.functional.cross_entropy(logits,labels)
    kd=nn.functional.kl_div(nn.functional.log_softmax(logits.float()/temperature,dim=-1),
        nn.functional.softmax(teacher_logits.float()/temperature,dim=-1),reduction="batchmean")
    return (1-alpha)*ce+alpha*temperature**2*kd

def install_multitask(model):
    from types import MethodType
    model.auxiliary_head=nn.Linear(model.config.hidden_size,2)
    model.sentiment_head=nn.Linear(model.config.hidden_size,3)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None):
        args=dict(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        if token_type_ids is not None: args["token_type_ids"]=token_type_ids
        h=self.base_model(**args).last_hidden_state[:,0]
        logits=self.sentiment_head(h)
        loss=None
        if labels is not None:
            # Auxiliary target: neutral versus subjective (derived from sentiment).
            aux=(labels!=0).long()
            loss=nn.functional.cross_entropy(logits,labels)+.3*nn.functional.cross_entropy(self.auxiliary_head(h),aux)
        return SequenceClassifierOutput(loss=loss,logits=logits)
    model.forward=MethodType(forward,model)
    return model
