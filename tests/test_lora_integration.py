
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from wan.utils import lora_utils

class TestLoraIntegration(unittest.TestCase):
    
    def test_apply_lora_called_i2v(self):
        """Test that apply_lora is called when lora_path is provided to WanI2V."""
        with patch('wan.image2video.WanModel') as MockModel, \
             patch('wan.image2video.WanI2V._configure_model') as mock_configure:
            
            # Mock the return value of configure_model to be a mock model
            mock_configure.return_value = MagicMock()
            
            from wan.image2video import WanI2V
            
            config = MagicMock()
            config.num_train_timesteps = 1000
            
            # Instantiate with lora_path
            pipeline = WanI2V(
                config=config,
                checkpoint_dir="/tmp",
                lora_path="/path/to/lora.safetensors"
            )
            
            # Check if _configure_model was called with lora_path
            # The args passed to _configure_model are: (model, use_sp, dit_fsdp, shard_fn, convert_model_dtype, lora_path)
            # or via kwargs. 
            
            # Since we mocked _configure_model, we just check if it was called with kwarg or arg
            call_args = mock_configure.call_args
            self.assertTrue(call_args is not None)
            
            # We expect lora_path to be passed. 
            # Depending on how it was called (positional or keyword), let's check.
            # keyword arg check
            if 'lora_path' in call_args.kwargs:
                self.assertEqual(call_args.kwargs['lora_path'], "/path/to/lora.safetensors")
            else:
                # If positional, it's the last argument.
                # args: (model, use_sp, dit_fsdp, shard_fn, convert_model_dtype, lora_path)
                # But wait, looking at the code:
                # self.low_noise_model = self._configure_model(...)
                # It is called with named arguments in the code I wrote?
                # Let's check the code:
                # self._configure_model(..., lora_path=lora_path)
                self.assertEqual(call_args.kwargs.get('lora_path'), "/path/to/lora.safetensors")

    def test_apply_lora_function(self):
        """Test the apply_lora utility function."""
        mock_model = MagicMock()
        mock_model.add_adapter = MagicMock()
        
        with patch('os.path.exists', return_value=True), \
             patch('wan.utils.lora_utils.load_file') as mock_load_file, \
             patch('wan.utils.lora_utils.set_peft_model_state_dict') as mock_set_state:
             
            # Setup mock state dict
            mock_load_file.return_value = {"state_dict": {}}
            
            lora_utils.apply_lora(mock_model, "dummy.safetensors")
            
            mock_model.add_adapter.assert_called()
            mock_set_state.assert_called()
            
if __name__ == '__main__':
    unittest.main()
