import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import sys

from .config import MODEL_CONFIGS
from . import core

class ModelManager:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "spksim"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_from_hub(self, model_name: str) -> core.SpeakerSimilarityModel:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")

        print(f"Models will be cached to: {self.cache_dir}", file=sys.stderr)
        config = MODEL_CONFIGS[model_name]
        
        model_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["onnx_filename"],
            cache_dir=self.cache_dir,
        )

        model_class = getattr(core, config["class"])
        
        if config["class"] == "WavLM":
            model_instance = model_class(
                model_path=model_path, 
                feature_extractor_repo=config["feature_extractor_repo"]
            )
        else:
            model_instance = model_class(model_path=model_path)
            
        return model_instance

    def get_model_from_local_paths(
        self, 
        model_type: str, 
        model_path: str, 
        feature_extractor_path: str = None
    ) -> core.SpeakerSimilarityModel:
        model_class_map = {
            "WavLM": core.WavLM,
            "WeSpeaker": core.WeSpeaker,
            "Resemblyzer": core.Resemblyzer,
        }
        
        model_class = model_class_map.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: '{model_type}'. Supported types: {list(model_class_map.keys())}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        if model_type == "WavLM":
            if not feature_extractor_path or not os.path.isdir(feature_extractor_path):
                raise ValueError("WavLM model requires a valid feature extractor directory path specified with --feature-extractor-path.")
            model_instance = model_class(
                model_path=model_path, 
                feature_extractor_repo=feature_extractor_path
            )
        else:
            model_instance = model_class(model_path=model_path)
            
        return model_instance