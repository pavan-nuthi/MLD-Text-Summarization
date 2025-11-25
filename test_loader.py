from utils.data_loader import load_bbc_data

try:
    # Point to the BBC News Summary directory
    # Note: The function expects the folder containing "News Articles" and "Summaries"
    # In our case, it is "BBC News Summary/BBC News Summary" or just "BBC News Summary"
    # Let's check the listing again.
    # Step 75: BBC News Summary/News Articles exists.
    # So the path is "BBC News Summary"
    
    ds = load_bbc_data("BBC News Summary")
    print("Successfully loaded dataset from local path!")
    print(ds)
    print(f"Train size: {len(ds['train'])}")
    print(f"Validation size: {len(ds['validation'])}")
    print(f"Test size: {len(ds['test'])}")
    print("Sample Document:", ds['train'][0]['document'][:100])
    print("Sample Summary:", ds['train'][0]['summary'][:100])
except Exception as e:
    print(f"Failed to load: {e}")
