import torch.nn as nn
from transformers import BartForConditionalGeneration, BartConfig

class BartSummarizer(nn.Module):
    def __init__(self, model_name='facebook/bart-base'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Remove arguments that BartForConditionalGeneration doesn't accept
        kwargs.pop('num_items_in_batch', None)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def freeze_encoder(self):
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False
