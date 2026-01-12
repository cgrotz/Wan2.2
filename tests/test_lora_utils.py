
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from wan.utils import lora_utils

class TestLoraUtils(unittest.TestCase):
    
    def test_apply_lora_returns_peft_model(self):
        """Test that apply_lora returns a model with adapters."""
        mock_model = nn.Linear(10, 10) # Simple model
        
        # Mock functions
        with patch('os.path.exists', return_value=True), \
             patch('wan.utils.lora_utils.load_file') as mock_load_file, \
             patch('wan.utils.lora_utils.get_peft_model') as mock_get_peft, \
             patch('wan.utils.lora_utils.set_peft_model_state_dict') as mock_set_state, \
             patch('torch.load') as mock_torch_load:
             
            # Setup mocks
            # If checking .bin
            mock_torch_load.return_value = {"state_dict": {}}
             
            # If checking .safetensors (default in code logic if ends with it, but here path doesn't)
            # let's pass a dummy path that doesn't end in safetensors to use torch.load
            
            mock_peft_model = MagicMock()
            mock_get_peft.return_value = mock_peft_model
            
            result_model = lora_utils.apply_lora(mock_model, "dummy.pt")
            
            # Assert get_peft_model was called
            mock_get_peft.assert_called()
            
            # Assert we returned the wrapped model
            self.assertEqual(result_model, mock_peft_model)
            
            # Assert state dict was set
            mock_set_state.assert_called()

if __name__ == '__main__':
    unittest.main()
