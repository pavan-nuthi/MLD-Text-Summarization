import argparse
import torch
from transformers import (
    Trainer, TrainingArguments, 
    BertTokenizer, BartTokenizer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np

from models.bertsum import BertSumExtractive
from models.bart import BartSummarizer
from utils.data_loader import load_bbc_data
from utils.preprocessing import preprocess_for_bertsum, preprocess_for_bart, get_oracle_ids

def train(args):
    # Load Data
    dataset = load_bbc_data(args.data_path)
    
    # Select Model & Tokenizer
    if args.model_type == 'bertsum':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertSumExtractive.from_pretrained('bert-base-uncased')
        
        def preprocess_function(examples):
            inputs = []
            all_labels = []
            all_cls_indices = []
            
            for doc, summary in zip(examples['document'], examples['summary']):
                # Preprocess
                processed = preprocess_for_bertsum(doc, tokenizer)
                
                # Generate Oracle Labels
                sentences = processed['sentences']
                labels = get_oracle_ids(sentences, summary, None)
                
                # Find CLS indices in input_ids
                # In our simple preprocess, we didn't explicitly track them, but we can find them.
                # However, our preprocess_for_bertsum flattened everything.
                # We need to be smarter there or reconstruct.
                
                # Let's fix preprocess_for_bertsum to return cls_indices or do it here.
                # Actually, let's just re-do the tokenization here properly for batching.
                
                # Re-implementation of robust tokenization for BERTSum:
                # 1. Tokenize sentences individually.
                # 2. Add [CLS] and [SEP].
                # 3. Concatenate.
                # 4. Track [CLS] indices.
                
                input_ids = []
                cls_indices = []
                
                for i, sent in enumerate(sentences):
                    # Encode
                    ids = tokenizer.encode(sent, add_special_tokens=True) # [CLS] ... [SEP]
                    # Remove [CLS] and [SEP] if we want to control placement?
                    # Standard BERT: [CLS] sent [SEP]
                    
                    cls_idx = len(input_ids) # The current position is where [CLS] will be
                    cls_indices.append(cls_idx)
                    
                    input_ids.extend(ids)
                
                # Truncate
                if len(input_ids) > 512:
                    input_ids = input_ids[:512]
                    # Ensure last is SEP? Not strictly necessary for this custom model but good practice.
                    if input_ids[-1] != tokenizer.sep_token_id:
                        input_ids[-1] = tokenizer.sep_token_id
                        
                # Filter cls_indices that are out of bounds
                cls_indices = [idx for idx in cls_indices if idx < len(input_ids)]
                labels = labels[:len(cls_indices)] # Truncate labels too
                
                inputs.append(input_ids)
                all_labels.append(labels)
                all_cls_indices.append(cls_indices)
            
            # Padding is handled by DataCollator? 
            # We need a custom collator for cls_indices and labels.
            return {
                "input_ids": inputs,
                "labels": all_labels,
                "cls_indices": all_cls_indices
            }
            
        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
        
        # Custom Collator
        def collate_fn(batch):
            max_len = max(len(x['input_ids']) for x in batch)
            max_sent = max(len(x['cls_indices']) for x in batch)
            
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            cls_indices = torch.full((len(batch), max_sent), -1, dtype=torch.long)
            labels = torch.full((len(batch), max_sent), -1, dtype=torch.float) # -1 for padding
            
            for i, item in enumerate(batch):
                seq_len = len(item['input_ids'])
                sent_len = len(item['cls_indices'])
                
                input_ids[i, :seq_len] = torch.tensor(item['input_ids'])
                attention_mask[i, :seq_len] = 1
                cls_indices[i, :sent_len] = torch.tensor(item['cls_indices'])
                labels[i, :sent_len] = torch.tensor(item['labels'])
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cls_indices": cls_indices,
                "labels": labels
            }
            
        data_collator = collate_fn

    elif args.model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartSummarizer('facebook/bart-base')
        model.freeze_encoder() # Freeze encoder to save memory
        
        def preprocess_function(examples):
            inputs = [doc for doc in examples['document']]
            targets = [sum for sum in examples['summary']]
            model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
            labels = tokenizer(targets, max_length=128, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model.model) # Use internal model for collator compatibility

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{args.model_type}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        save_safetensors=False,  # tied embeddings (e.g., BART) cannot be losslessly serialized with safetensors
        remove_unused_columns=False if args.model_type == 'bertsum' else True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adafactor"
    )

    from utils.callbacks import MemoryCleanupCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        callbacks=[MemoryCleanupCallback()]
    )

    trainer.train()
    trainer.save_model(f"./saved_models/{args.model_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['bertsum', 'bart'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    train(args)
