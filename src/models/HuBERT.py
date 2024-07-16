import torch.nn as nn
from transformers import AutoModelForAudioClassification

class HuBERT(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(HuBERT, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits

def get_model():
    return HuBERT("superb/hubert-base-superb-ks", num_labels=2)