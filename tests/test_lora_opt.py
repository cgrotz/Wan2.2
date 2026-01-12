
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from wan.utils import lora_utils

class TestLoraOptimization(unittest.TestCase):
    
    def test_apply_lora_merges_weights(self):
        """Test that apply_lora calls merge_and_unload when merge_lora=True."""
        mock_model = nn.Linear(10, 10) 
        
        with patch('os.path.exists', return_value=True), \
             patch('wan.utils.lora_utils.load_file') as mock_load_file, \
             patch('wan.utils.lora_utils.get_peft_model') as mock_get_peft, \
             patch('wan.utils.lora_utils.set_peft_model_state_dict') as mock_set_state, \
             patch('torch.load') as mock_torch_load:
             
            mock_torch_load.return_value = {"state_dict": {}}
            
            # Setup mock peft model
            mock_peft_model = MagicMock()
            mock_get_peft.return_value = mock_peft_model
            
            # Setup merged model return
            mock_merged_model = MagicMock()
            mock_peft_model.merge_and_unload.return_value = mock_merged_model
            
            # Call with merge_lora=True (default)
            result_model = lora_utils.apply_lora(mock_model, "dummy.pt", merge_lora=True)
            
            # Assert merge_and_unload was called
            mock_peft_model.merge_and_unload.assert_called()
            
            # Assert we returned the merged model, not the peft wrapper
            self.assertEqual(result_model, mock_merged_model)

    def test_apply_lora_no_merge(self):
        """Test that apply_lora does NOT call merge_and_unload when merge_lora=False."""
        mock_model = nn.Linear(10, 10) 
        
        with patch('os.path.exists', return_value=True), \
             patch('wan.utils.lora_utils.load_file'), \
             patch('wan.utils.lora_utils.get_peft_model') as mock_get_peft, \
             patch('wan.utils.lora_utils.set_peft_model_state_dict'), \
             patch('torch.load') as mock_torch_load:
             
            mock_torch_load.return_value = {"state_dict": {}}
            
            mock_peft_model = MagicMock()
            mock_get_peft.return_value = mock_peft_model
            
            result_model = lora_utils.apply_lora(mock_model, "dummy.pt", merge_lora=False)
            
            mock_peft_model.merge_and_unload.assert_not_called()
            self.assertEqual(result_model, mock_peft_model)

if __name__ == '__main__':
    unittest.main()
