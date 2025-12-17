# spksim/manager.py

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from .config import MODEL_CONFIGS
from . import core

class ModelManager:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "spksim"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_name: str) -> core.SpeakerSimilarityModel:
        """统一的模型加载入口"""

        # 1. 查表：看看这个名字在不在我们的配置里
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型名称: '{model_name}'。请在 config.py 中定义该模型。")

        config = MODEL_CONFIGS[model_name]
        model_class_type = config["class"]  # 从配置中获取类型 (WavLM/WeSpeaker/...)

        # 2. 准备模型路径
        if config.get("source") == "local":
            # --- 情况 A: 本地模型 ---
            model_path = config["path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"配置文件中指定的本地模型不存在: {model_path}")
            print(f"Loading local model: {model_name} ({model_class_type})", file=sys.stderr)

        else:
            # --- 情况 B: 远程模型 (HuggingFace) ---
            print(f"Downloading/Loading remote model: {model_name}", file=sys.stderr)
            model_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["onnx_filename"],
                cache_dir=self.cache_dir,
            )

        # 3. 实例化模型类
        # 动态获取类，例如 core.WavLM, core.WeSpeaker
        model_class = getattr(core, model_class_type)

        if model_class_type == "WavLM":
            # WavLM 需要额外的 feature_extractor配置
            # 如果是本地模型，config里应该配一个 'feature_extractor_path'
            # 如果是远程模型，用 'feature_extractor_repo'
            fe_path = config.get("feature_extractor_path") or config.get("feature_extractor_repo")

            model_instance = model_class(
                model_path=model_path,
                feature_extractor_repo=fe_path
            )
        else:
            model_instance = model_class(model_path=model_path)

        return model_instance