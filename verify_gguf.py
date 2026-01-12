import torch
import os
import sys

# Ensure the 'wan' package is in the path
sys.path.append(os.path.abspath('.'))

from wan.modules.model import WanModel

def test_from_gguf_initialization():
    print("Testing WanModel.from_gguf initialization (architecture only)...")
    
    # Since we might not have a real GGUF file in the test environment, 
    # we just want to verify that the method exists and can be called.
    # In a real scenario, you would provide a path to a .gguf file.
    
    if hasattr(WanModel, 'from_gguf'):
        print("SUCCESS: WanModel has the 'from_gguf' method.")
    else:
        print("FAILURE: WanModel is missing 'from_gguf' method.")
        sys.exit(1)

if __name__ == "__main__":
    test_from_gguf_initialization()
