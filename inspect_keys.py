import torch
import sys

path = sys.argv[1]
state_dict = torch.load(path, map_location="cpu")
print("Keys sample:", list(state_dict.keys())[:5])
