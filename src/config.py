"""Utility functions for loading and managing configuration."""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_device(config: Dict[str, Any] = None) -> str:
    """
    Get the device to use for computation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda', 'cpu', or 'mps')
    """
    import torch
    
    if config and 'device' in config:
        device = config['device']
        if device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif device == 'mps' and torch.backends.mps.is_available():
            return 'mps'
    
    # Auto-detect
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
