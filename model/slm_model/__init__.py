"""slm_model — a from-scratch small language model for candidate relevance + resume extraction.

The package root stays torch-free so lightweight consumers (the data pipeline) can import
`slm_model.prompting` without pulling torch in. Import heavy modules explicitly:

    from slm_model.config import ModelConfig, PRESETS
    from slm_model.model import SLM
    from slm_model.engine import Engine
"""
__version__ = "0.1.0"
