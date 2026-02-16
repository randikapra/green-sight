"""
Configuration loader and manager.
Loads YAML config and provides dot-notation access.
"""

import yaml
import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from typing import Any, Optional


class Config:
    """Hierarchical config with dot-notation access."""
    
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # Keep dicts with non-string keys as plain dicts
                if any(not isinstance(k, str) for k in value.keys()):
                    setattr(self, str(key), value)
                else:
                    setattr(self, str(key), Config(value))
            elif isinstance(value, list):
                # Convert list of dicts to list of Configs (if all-string keys)
                converted = []
                for item in value:
                    if isinstance(item, dict) and all(isinstance(k, str) for k in item.keys()):
                        converted.append(Config(item))
                    else:
                        converted.append(item)
                setattr(self, str(key), converted)
            else:
                setattr(self, str(key), value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item 
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(config_path: str = "configs/default.yaml", overrides: Optional[dict] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: path to YAML config file
        overrides: dict of overrides (e.g., from CLI args)
    
    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        data = _deep_merge(data, overrides)
    
    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_device(cfg: Config) -> torch.device:
    """Resolve device from config."""
    device_str = cfg.project.device
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment(cfg: Config) -> torch.device:
    """
    Full environment setup: seed, device, directories.
    Call this at the start of every script.
    """
    set_seed(cfg.project.seed)
    device = resolve_device(cfg)
    
    # Create directories
    for dir_path in [
        cfg.data.raw_dir, 
        cfg.data.processed_dir,
        cfg.project.output_dir, 
        cfg.project.log_dir
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"  {cfg.project.name}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Seed: {cfg.project.seed}")
    print(f"{'='*60}")
    
    return device


def get_class_names(cfg: Config) -> list:
    """Extract ordered class names from config."""
    classes = cfg.preprocessing.classes
    if isinstance(classes, Config):
        classes = classes.to_dict()
    return [classes[i]['name'] for i in sorted(classes.keys())]


def get_class_colors(cfg: Config) -> list:
    """Extract ordered class colors from config."""
    classes = cfg.preprocessing.classes
    if isinstance(classes, Config):
        classes = classes.to_dict()
    return [classes[i]['color'] for i in sorted(classes.keys())]


def get_enabled_models(cfg: Config) -> dict:
    """Return dict of enabled model configs."""
    models = cfg.models.to_dict()
    return {name: conf for name, conf in models.items() if conf.get('enabled', False)}


def resolve_periods(cfg: Config) -> list:
    """Safely extract periods as list of plain dicts."""
    periods = cfg.data.periods
    if isinstance(periods, list):
        return [p.to_dict() if isinstance(p, Config) else p for p in periods]
    if isinstance(periods, Config):
        return list(periods.to_dict().values())
    return periods


def resolve_classes(cfg: Config, city_key: str = None) -> dict:
    """Safely extract classes as dict with int keys and plain dict values.
    If city_key is given, use city-specific classes from study_areas."""
    if city_key is not None:
        areas = cfg.study_areas
        if isinstance(areas, Config):
            areas = areas.to_dict()
        city = areas[city_key]
        classes = city.get('classes', {})
    else:
        classes = cfg.preprocessing.classes if hasattr(cfg.preprocessing, 'classes') else {}
        if isinstance(classes, Config):
            classes = classes.to_dict()
    return {int(k): (v.to_dict() if isinstance(v, Config) else v) for k, v in classes.items()}


def get_study_areas(cfg: Config) -> dict:
    """Return dict of study area configs."""
    areas = cfg.study_areas
    if isinstance(areas, Config):
        areas = areas.to_dict()
    return areas


def get_class_names_for_city(cfg: Config, city_key: str) -> list:
    """Get ordered class names for a specific city."""
    classes = resolve_classes(cfg, city_key)
    return [classes[i]['name'] for i in sorted(classes.keys())]


def get_class_colors_for_city(cfg: Config, city_key: str) -> list:
    """Get ordered class colors for a specific city."""
    classes = resolve_classes(cfg, city_key)
    return [classes[i]['color'] for i in sorted(classes.keys())]
