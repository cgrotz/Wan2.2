
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

class TestDualLoraIntegration(unittest.TestCase):
    
    def test_dual_lora_application(self):
        """Test that lora_path and lora_path_high are applied to correct models in WanI2V."""
        with patch('wan.image2video.WanModel') as MockModel, \
             patch('wan.image2video.WanI2V._configure_model') as mock_configure, \
             patch('wan.image2video.Wan2_1_VAE') as MockVAE, \
             patch('wan.image2video.T5EncoderModel') as MockT5:

            # Setup mocks
            MockModel.from_pretrained.return_value = MagicMock()
            mock_configure.side_effect = lambda model, **kwargs: model # Return the model passed in
            
            from wan.image2video import WanI2V
            
            config = MagicMock()
            config.num_train_timesteps = 1000
            
            # Instantiate with both LoRA paths
            pipeline = WanI2V(
                config=config,
                checkpoint_dir="/tmp",
                lora_path="/path/to/low_noise.safetensors",
                lora_path_high="/path/to/high_noise.safetensors"
            )
            
            # Check _configure_model calls
            # It should be called twice: once for low noise, once for high noise
            self.assertEqual(mock_configure.call_count, 2)
            
            # Verify arguments for each call
            # We expect one call with lora_path="/path/to/low_noise.safetensors"
            # and one with lora_path="/path/to/high_noise.safetensors" (passed as lora_path arg name)
            
            call_args_list = mock_configure.call_args_list
            
            # Extract lora_path arguments from calls
            lora_paths_used = []
            for call in call_args_list:
                # _configure_model signature: (model, use_sp, dit_fsdp, shard_fn, convert_model_dtype, lora_path=None)
                # It might be passed as positional or keyword
                if 'lora_path' in call.kwargs:
                     lora_paths_used.append(call.kwargs['lora_path'])
                else:
                    # If positional, it's the last one
                    # We can't easily guess positional args from the mock without inspecting all of them
                    # But based on our implementation:
                    # self.low_noise_model = self._configure_model(..., lora_path=lora_path)
                    # self.high_noise_model = self._configure_model(..., lora_path=lora_path_high)
                    pass

            self.assertIn("/path/to/low_noise.safetensors", lora_paths_used)
            self.assertIn("/path/to/high_noise.safetensors", lora_paths_used)

if __name__ == '__main__':
    unittest.main()
