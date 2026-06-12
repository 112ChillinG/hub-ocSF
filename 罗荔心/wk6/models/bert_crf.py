# models/bert_crf.py

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BertCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs.last_hidden_state)  # [bs, seq, num_labels]

        loss = None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
        return {"loss": loss, "emissions": emissions}

    def decode(self, input_ids, attention_mask):
        emissions = self.forward(input_ids, attention_mask)["emissions"]
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        return predictions