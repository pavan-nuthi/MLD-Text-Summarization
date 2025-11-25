import re
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def clean_text(text):
    """
    Cleans text by removing HTML tags and extra whitespace.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_sentences(text):
    """
    Segments text into a list of sentences using NLTK.
    """
    return nltk.sent_tokenize(text)

def preprocess_for_bertsum(text, tokenizer, max_length=512):
    """
    Prepares text for BERTSum by adding [CLS] and [SEP] tokens between sentences.
    Returns tokenized input_ids and attention_mask.
    """
    sentences = segment_sentences(clean_text(text))
    
    # BERTSum expects [CLS] sent1 [SEP] [CLS] sent2 [SEP] ...
    # We will construct the text manually then tokenize, or tokenize sentences and join.
    # A simpler approach for standard BERT models used as BERTSum:
    # Join with [SEP] [CLS] but standard BERT might not handle multiple CLS well without custom attention.
    # The standard BERTSum implementation usually modifies the input embedding or just uses [CLS] at start of each sentence.
    
    # For this implementation, we will format as:
    # [CLS] sent1 [SEP] [CLS] sent2 [SEP] ...
    # And we'll need to handle segment embeddings if we want strictly faithful BERTSum, 
    # but for a basic version, we can just treat it as a long sequence.
    
    # However, standard HuggingFace BertTokenizer adds [CLS] at start and [SEP] at end.
    # We want to keep sentences separate.
    
    tokenized_sentences = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    
    # Flatten
    input_ids = []
    for s in tokenized_sentences:
        input_ids.extend(s)
    
    # Truncate
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        # Ensure it ends with SEP if we cut it off? 
        # Actually, let's just keep it simple.
        if input_ids[-1] != tokenizer.sep_token_id:
             input_ids[-1] = tokenizer.sep_token_id

    attention_mask = [1] * len(input_ids)
    
    # Padding
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentences": sentences # Return original sentences for reference/extraction
    }

def preprocess_for_bart(text, summary, tokenizer, max_source_length=128, max_target_length=64):
    """
    Prepares text and summary for BART.
    """
    clean_src = clean_text(text)
    clean_tgt = clean_text(summary)
    
    model_inputs = tokenizer(
        clean_src, 
        max_length=max_source_length, 
        padding="max_length", 
        truncation=True
    )
    
    labels = tokenizer(
        clean_tgt, 
        max_length=max_target_length, 
        padding="max_length", 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_oracle_ids(sentences, summary, rouge_metric):
    """
    Greedily selects sentences from the source that maximize ROUGE-L score with the summary.
    Returns a list of 0s and 1s indicating selected sentences.
    """
    # This is a simplified version. A full implementation requires calculating ROUGE for every combination.
    # We will use a simple heuristic: if a sentence has high overlap with summary, keep it.
    # Or better, use the standard greedy approach.
    
    selected_indices = []
    current_summary = ""
    
    # We need to split summary into sentences for better comparison or just treat as blob.
    # Let's try to match source sentences to summary content.
    
    # Optimization: Pre-calculate rouge scores for each sentence against the summary?
    # That's expensive.
    
    # Simple heuristic:
    # 1. Tokenize summary and source sentences.
    # 2. For each source sentence, calculate overlap (e.g. ROUGE-1 recall) with summary.
    # 3. Select top-k sentences or those above a threshold.
    
    # Let's use a simple overlap coefficient for speed in this demo.
    
    summary_tokens = set(clean_text(summary).split())
    
    scores = []
    for i, sent in enumerate(sentences):
        sent_tokens = set(clean_text(sent).split())
        if len(sent_tokens) == 0:
            scores.append(0)
            continue
        
        overlap = len(summary_tokens.intersection(sent_tokens))
        # Normalize by sentence length to favor concise sentences? 
        # Or just raw overlap count to maximize coverage?
        # Usually we want to maximize ROUGE, which is recall-oriented.
        score = overlap / len(summary_tokens) if len(summary_tokens) > 0 else 0
        scores.append(score)
        
    # Select top 3 sentences (typical for news)
    # Or use a threshold.
    
    # Let's pick top 3
    top_k = 3
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    labels = [0] * len(sentences)
    for idx in ranked_indices:
        if scores[idx] > 0.1: # Minimum relevance threshold
            labels[idx] = 1
            
    return labels
