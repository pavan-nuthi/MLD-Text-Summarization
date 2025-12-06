import os
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def load_bbc_data(data_path=None):
    """
    Loads the BBC News Summary dataset.
    If data_path is provided, tries to load from local files.
    Otherwise, tries to load from HuggingFace Hub (gopalkalpande/bbc-news-summary).
    
    Returns a DatasetDict with train, validation, and test splits.
    """
    if data_path is None and os.path.exists("BBC News Summary"):
        data_path = "BBC News Summary"

    if data_path and os.path.exists(data_path):
        print(f"Loading from local path: {data_path}")
        
        # Check for News Articles and Summaries folders
        articles_path = os.path.join(data_path, "News Articles")
        summaries_path = os.path.join(data_path, "Summaries")
        
        if not os.path.exists(articles_path) or not os.path.exists(summaries_path):
            # Try checking if they are inside a subdirectory (some unzips create nested folders)
            # Based on file listing: BBC News Summary/BBC News Summary/News Articles might exist?
            # But user listed BBC News Summary/News Articles directly.
            # Let's handle the nested case just in case, or just fail if not found.
            
            # Check if user passed the parent folder and the subfolders are deeper
            # But let's assume the user passes the folder containing "News Articles" and "Summaries"
            raise ValueError(f"Could not find 'News Articles' and 'Summaries' directories in {data_path}")

        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        
        documents = []
        summaries = []
        filenames = []
        
        for category in categories:
            cat_articles_path = os.path.join(articles_path, category)
            cat_summaries_path = os.path.join(summaries_path, category)
            
            if not os.path.exists(cat_articles_path):
                print(f"Warning: Category {category} not found in {articles_path}")
                continue
                
            files = os.listdir(cat_articles_path)
            
            for file in files:
                if not file.endswith('.txt'):
                    continue
                    
                # Read Article
                with open(os.path.join(cat_articles_path, file), 'r', encoding='utf-8', errors='replace') as f:
                    documents.append(f.read())
                    
                # Read Summary
                sum_file = os.path.join(cat_summaries_path, file)
                if os.path.exists(sum_file):
                    with open(sum_file, 'r', encoding='utf-8', errors='replace') as f:
                        summaries.append(f.read())
                else:
                    # Should not happen in this dataset usually
                    print(f"Warning: Summary not found for {file}")
                    # Remove the last document to keep aligned
                    documents.pop()
                    continue
                
                # Store filename (category/file)
                filenames.append(f"{category}/{file}")
        
        # Create Dataset
        data = {'document': documents, 'summary': summaries, 'filename': filenames}
        full_dataset = Dataset.from_dict(data)
        
        # Split 80/10/10
        train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        
        final_dataset = DatasetDict({
            'train': train_testvalid['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
        
        return final_dataset

    try:
        print("Attempting to load from HuggingFace Hub: gopalkalpande/bbc-news-summary")
        dataset = load_dataset("gopalkalpande/bbc-news-summary")
        # The dataset usually comes with 'train' split only or similar.
        # We need to inspect it. Assuming it has 'File_path', 'Articles', 'Summaries' columns.
        
        # If the dataset structure is different, we might need to adapt.
        # Let's assume it returns a Dataset object.
        
        # Standardize column names
        # We need 'document' and 'summary'
        
        # Note: gopalkalpande/bbc-news-summary might have columns like 'Articles', 'Summaries'
        
        # Let's do a quick check or just map generic names
        if 'train' in dataset:
            full_data = dataset['train']
        else:
            full_data = dataset
            
        # Rename columns if necessary
        if 'Articles' in full_data.column_names:
            full_data = full_data.rename_column("Articles", "document")
        if 'Summaries' in full_data.column_names:
            full_data = full_data.rename_column("Summaries", "summary")
            
        # Split 80/10/10
        # First split train (80%) and temp (20%)
        train_testvalid = full_data.train_test_split(test_size=0.2, seed=42)
        
        # Split temp into valid (50% of 20% = 10%) and test (50% of 20% = 10%)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        
        final_dataset = DatasetDict({
            'train': train_testvalid['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
        
        return final_dataset

    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Please ensure you have internet access or provide a local dataset path.")
        raise e

if __name__ == "__main__":
    # Test loading
    ds = load_bbc_data()
    print("Dataset loaded successfully")
    print(ds)
