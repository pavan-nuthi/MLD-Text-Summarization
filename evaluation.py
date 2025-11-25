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
    test_data = dataset['test']
    
    rouge = load("rouge")
    bertscore = load("bertscore")
    
    predictions = []
    references = test_data['summary']
    
    if args.model_type == 'bertsum':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertSumExtractive.from_pretrained(args.model_path).to(device)
        model.eval()
        
        for item in tqdm(test_data):
            doc = item['document']
            sentences = segment_sentences(clean_text(doc))
            
            # Prepare input
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
                
            # Select top-k or threshold
            # Let's pick top 3
            scores = probs[0].cpu().numpy()
            top_k_indices = scores.argsort()[-3:][::-1]
            top_k_indices.sort() # Restore original order
            
            selected_sentences = [sentences[i] for i in top_k_indices if i < len(sentences)]
            summary = " ".join(selected_sentences)
            predictions.append(summary)
            
    elif args.model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # Initialize with base model structure
        model = BartSummarizer('facebook/bart-base').to(device)
        
        # Load state dict manually because Trainer saved the wrapper
        try:
            state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded fine-tuned BART model weights.")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Falling back to default initialization (this will likely yield poor results).")
            
        model.eval()
        
        for item in tqdm(test_data):
            doc = item['document']
            inputs = tokenizer(doc, max_length=1024, truncation=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=128, early_stopping=True)
                
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            predictions.append(summary)

    # Compute Metrics
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE Results:", rouge_results)
    
    # BERTScore
    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(f"BERTScore F1: {np.mean(bertscore_results['f1'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['bertsum', 'bart'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    
    evaluate(args)
