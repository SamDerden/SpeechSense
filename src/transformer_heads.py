import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from types import MethodType

def install_head(model, technique):
    hidden = model.config.hidden_size
    model.experiment_head = nn.Sequential(nn.Dropout(.2), nn.Linear(hidden*(2 if technique=="mean_pool" else 1), hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden,3))
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,inputs_embeds=None,labels=None,**kwargs):
        args = dict(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds,return_dict=True)
        if token_type_ids is not None:
            args["token_type_ids"]=token_type_ids
        h=self.base_model(**args).last_hidden_state
        x=h[:,0]
        if technique=="mean_pool":
            mask=attention_mask.unsqueeze(-1).to(h.dtype)
            x=torch.cat([x,(h*mask).sum(1)/mask.sum(1).clamp_min(1)],dim=-1)
        return SequenceClassifierOutput(logits=self.experiment_head(x))
    model.forward=MethodType(forward,model)
    return model
