"""Configuration management for the statistical analysis platform."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the platform."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from yaml file and environment variables."""
        load_dotenv()

        # Find config file
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._default_config()

        # Override with environment variables
        self._config['llm']['api_key'] = os.getenv('ANTHROPIC_API_KEY', '')

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'llm': {
                'provider': 'anthropic',
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 4096,
                'temperature': 0.1
            },
            'statistics': {
                'significance_level': 0.05,
                'confidence_level': 0.95,
                'normality_test_threshold': 50,
                'min_sample_size_parametric': 30
            },
            'visualization': {
                'default_style': 'seaborn-v0_8-whitegrid',
                'figure_dpi': 300,
                'figure_format': 'png',
                'color_palette': 'colorblind'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def significance_level(self) -> float:
        return self.get('statistics.significance_level', 0.05)

    @property
    def confidence_level(self) -> float:
        return self.get('statistics.confidence_level', 0.95)

    @property
    def llm_model(self) -> str:
        return self.get('llm.model', 'claude-sonnet-4-20250514')

    @property
    def api_key(self) -> str:
        return self.get('llm.api_key', '')
