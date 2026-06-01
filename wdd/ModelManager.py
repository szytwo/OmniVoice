import threading

import torch

from omnivoice import OmniVoice
from wdd.file_utils import logging


class ModelManager:
    def __init__(
        self, device: str = "cuda:0", dtype: str = "float16", load_asr: bool = False
    ):
        self.device = device
        self.dtype = dtype
        self.load_asr = load_asr
        self.models = {
            "OmniVoice": None,
        }
        self.locks = {
            "OmniVoice": threading.Lock(),
        }

    @staticmethod
    def _dtype_from_str(s: str) -> torch.dtype:
        s = (s or "").strip().lower()
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp16", "float16", "half"):
            return torch.float16
        if s in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")

    def _load_model(self, model_type: str):
        """
        内部方法：加载指定类型的模型。
        """
        logging.info(f"Loading model: {model_type}")

        if model_type == "OmniVoice":
            model_dir = "checkpoints/OmniVoice"
            asr_model_name = "checkpoints/whisper-large-v2"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        dtype = self._dtype_from_str(self.dtype)

        return OmniVoice.from_pretrained(
            model_dir,
            device_map=self.device,
            dtype=dtype,
            load_asr=self.load_asr,
            asr_model_name=asr_model_name,
        )

    def get_model(self, model_type: str):
        """
        获取指定类型的模型实例，按需加载，确保线程安全。
        """
        logging.info(f"get_model: {model_type}")
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 常驻模式：双重检查锁定，确保线程安全加载并缓存实例
        if self.models[model_type] is None:
            with self.locks[model_type]:  # 确保线程安全
                if self.models[model_type] is None:  # 双重检查锁定
                    omnivoice_model = self._load_model(model_type)
                    self.models[model_type] = omnivoice_model

        return self.models[model_type]
