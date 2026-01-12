
import unittest
from unittest.mock import MagicMock
import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())

# Mock diffusers dependencies if they are not installed in the test env, 
# but assuming they are available since we are importing them in the code.
# However, to be safe and fast, we can try to import the actual classes.

try:
    from wan.modules.model import WanModel
except ImportError:
    # If environment is broken (like missing torch), we might fail here.
    # But we want to test the attribute existence.
    pass

class TestWanModelAdapter(unittest.TestCase):
    
    def test_has_add_adapter(self):
        """Test that WanModel has add_adapter method."""
        # We can't easily instantiate WanModel without full config/weights usually,
        # but we can check the class attribute or try to instantiate with mocks if needed.
        # Checking class attribute is safer for a mixin method.
        
        from wan.modules.model import WanModel
        self.assertTrue(hasattr(WanModel, 'add_adapter'), "WanModel should have add_adapter method")
        self.assertTrue(callable(getattr(WanModel, 'add_adapter')), "add_adapter should be callable")

if __name__ == '__main__':
    unittest.main()
