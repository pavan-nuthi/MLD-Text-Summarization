import torch
import gc
from transformers import TrainerCallback

class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
