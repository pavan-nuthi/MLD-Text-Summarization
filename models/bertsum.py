import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertSumExtractive(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Inter-sentence Transformer (Summarization Layer)
        # We can use a simple TransformerEncoder or just a Linear layer for simplicity if compute is limited.
        # The proposal mentions "Inter-sentence Transformer".
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # Classification layer
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, cls_indices=None, labels=None):
        """
        cls_indices: Tensor of shape (batch_size, num_sentences) containing indices of [CLS] tokens.
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Extract [CLS] tokens for each sentence
        # cls_indices shape: (batch_size, max_sentences)
        # We need to gather the embeddings.
        
        if cls_indices is None:
            # Fallback: assume only one CLS at 0 (standard BERT) - not useful for extractive sum of multiple sentences
            # Or we can try to find 101 (CLS ID) in input_ids
            raise ValueError("cls_indices must be provided for BertSum")

        batch_size = input_ids.size(0)
        max_sentences = cls_indices.size(1)
        hidden_size = sequence_output.size(2)
        
        # Gather embeddings
        # We need to flatten batch and gather, or use gather on specific dim
        # sequence_output: [B, L, H]
        # cls_indices: [B, S]
        
        # Expand cls_indices to [B, S, H] to gather
        # This is tricky. Let's loop for simplicity or use efficient gather.
        
        batch_cls_vectors = []
        for i in range(batch_size):
            # Get indices for this batch item
            indices = cls_indices[i] # [S]
            # Filter out padding (-1 or 0 if 0 is not CLS)
            # We assume valid indices are >= 0.
            # However, 0 is usually the first CLS.
            
            # Gather
            # We need to handle padding in cls_indices (e.g. -1)
            valid_indices = indices[indices >= 0]
            
            if len(valid_indices) == 0:
                 # Should not happen if data is correct
                 vectors = torch.zeros(1, hidden_size).to(input_ids.device)
            else:
                vectors = sequence_output[i, valid_indices, :] # [Num_Valid_S, H]
            
            # Pad back to max_sentences
            if vectors.size(0) < max_sentences:
                pad_size = max_sentences - vectors.size(0)
                pad = torch.zeros(pad_size, hidden_size).to(input_ids.device)
                vectors = torch.cat([vectors, pad], dim=0)
            elif vectors.size(0) > max_sentences:
                vectors = vectors[:max_sentences]
                
            batch_cls_vectors.append(vectors)
            
        sentence_embeddings = torch.stack(batch_cls_vectors) # [B, S, H]
        
        # Pass through Inter-sentence Transformer
        # Transformer expects [Seq, Batch, Dim] by default, but batch_first=False default.
        # We have [B, S, H]. Let's transpose.
        sentence_embeddings = sentence_embeddings.permute(1, 0, 2) # [S, B, H]
        
        encoded_sentences = self.transformer_encoder(sentence_embeddings)
        
        encoded_sentences = encoded_sentences.permute(1, 0, 2) # [B, S, H]
        
        logits = self.classifier(encoded_sentences).squeeze(-1) # [B, S]
        probs = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            # labels: [B, S] float tensor of 0s and 1s
            # Mask out padding in labels?
            # We need to ensure labels match the shape of logits.
            # And handle padding.
            
            loss_fct = nn.BCELoss()
            
            # Create mask for valid sentences (not padded)
            # We can infer from cls_indices where it is not padding?
            # Or just assume labels has -100 for ignore index?
            # BCELoss doesn't support ignore_index.
            
            # Let's assume labels are 0 or 1, and we mask manually.
            # We can use the fact that padded sentences have 0 embeddings or we can pass a mask.
            
            # Simple way: use attention mask for sentences?
            # We don't have explicit sentence mask passed in.
            # But we can infer it from cls_indices (if we kept padding info).
            
            # Let's just flatten and compute loss, assuming labels are correct (0 for padding is fine if we want to suppress them, 
            # but better to ignore them).
            
            # If we used -1 in cls_indices for padding, we can use that.
            # In the loop above:
            # valid_indices = indices[indices >= 0]
            # We padded vectors with 0.
            
            # Let's assume labels have -1 for padding.
            active_loss = labels.view(-1) != -1
            active_logits = probs.view(-1)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            loss = loss_fct(active_logits, active_labels.float())
            
        return {"loss": loss, "logits": probs} if loss is not None else probs
