"""
model_registry.py

This file handles the model registry for the AI tutor.
It provides a simple interface to get model paths from the config file.

"""
from config_loader import get_model_registry

class ModelRegistry:
    """Simple model registry for model paths"""
    
    def __init__(self):
        """Initializes the ModelRegistry by loading the registry from the config file."""
        self.registry = get_model_registry()

    def get_model_path(self, model_type, model_name):
        """Get model path with basic validation"""
        if model_type not in self.registry:
            print(f"Error: Unknown model type: '{model_type}'")
            return None
        
        if model_name not in self.registry[model_type]:
            print(f"Error: Unknown model: '{model_name}' for type '{model_type}'")
            return None
        
        return self.registry[model_type][model_name]
