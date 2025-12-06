import argparse
import torch
from transformers import BertTokenizer, BartTokenizer
from evaluate import load
from tqdm import tqdm
import numpy as np

from models.bertsum import BertSumExtractive
from models.bart import BartSummarizer
from utils.data_loader import load_bbc_data
from utils.preprocessing import segment_sentences, clean_text

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_bbc_data(args.data_path)
    
    if args.split == 'all':
        splits_to_run = ['train', 'validation', 'test']
    else:
        splits_to_run = [args.split]
        
    print(f"Running evaluation on splits: {splits_to_run}")

    rouge = load("rouge")
    bertscore = load("bertscore")
    
    # Initialize Model & Tokenizer once
    if args.model_type == 'bertsum':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertSumExtractive.from_pretrained(args.model_path).to(device)
    elif args.model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartSummarizer('facebook/bart-base').to(device)
        try:
            state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded fine-tuned BART model weights.")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Falling back to default initialization.")
            
    model.eval()

    for split in splits_to_run:
        print(f"Processing split: {split}")
        data_split = dataset[split]
        
        predictions = []
        references = data_split['summary']
        filenames = data_split['filename'] if 'filename' in data_split.column_names else [f"{split}_{i}.txt" for i in range(len(data_split))]
        
        for item in tqdm(data_split):
            doc = item['document']
            
            if args.model_type == 'bertsum':
                from utils.preprocessing import segment_sentences, clean_text
                sentences = segment_sentences(clean_text(doc))
                
                input_ids = []
                cls_indices = []
                for sent in sentences:
                    ids = tokenizer.encode(sent, add_special_tokens=True)
                    cls_idx = len(input_ids)
                    cls_indices.append(cls_idx)
                    input_ids.extend(ids)
                    
                if len(input_ids) > 512:
                    input_ids = input_ids[:512]
                
                cls_indices = [idx for idx in cls_indices if idx < len(input_ids)]
                
                input_tensor = torch.tensor([input_ids]).to(device)
                cls_tensor = torch.tensor([cls_indices]).to(device)
                
                with torch.no_grad():
                    probs = model(input_tensor, cls_indices=cls_tensor)
                    
                scores = probs[0].cpu().numpy()
                top_k_indices = scores.argsort()[-3:][::-1]
                top_k_indices.sort()
                
                selected_sentences = [sentences[i] for i in top_k_indices if i < len(sentences)]
                summary = " ".join(selected_sentences)
                predictions.append(summary)
                
            elif args.model_type == 'bart':
                inputs = tokenizer(doc, max_length=1024, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=128, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                predictions.append(summary)

        # Compute Metrics (Optional per split, but good to see)
        print(f"--- Metrics for {split} ---")
        rouge_results = rouge.compute(predictions=predictions, references=references)
        print("ROUGE Results:", rouge_results)
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
        print(f"BERTScore F1: {np.mean(bertscore_results['f1'])}")

        # Save summaries
        output_dir = f"generated_summaries_{args.model_type}"
        import os
        
        for pred, filename in zip(predictions, filenames):
            # filename is like "business/001.txt"
            # We want output_dir/business/001.txt
            
            # Handle if filename is just a name or has path
            # User wants: business/001.txt -> business/summary_001.txt
            
            dirname, basename = os.path.split(filename)
            new_basename = f"summary_{basename}"
            save_path = os.path.join(output_dir, dirname, new_basename)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, "w") as f:
                f.write(pred)
                
        print(f"Saved {len(predictions)} summaries to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['bertsum', 'bart'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default='all', choices=['train', 'validation', 'test', 'all'], help="Split to evaluate on")
    args = parser.parse_args()
    
    evaluate(args)
