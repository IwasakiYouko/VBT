"""应用配置：负责持久化设置的读写与旧版本迁移。

将配置从散落在 UI 中的字段集中到一个 dataclass，
使 main.py 只需关心「界面变量 <-> 配置对象」的映射。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class AppConfig:
    """程序的持久化配置，全部以内部「编码值」而非界面标签保存。"""

    ffmpeg_path: Optional[str] = None
    output_dir: Optional[str] = None
    remove_audio: bool = False
    crop_9x16: bool = True
    remove_after_export: bool = False
    encoder: str = "auto"
    blur_method: str = "gaussian"
    blur_strength: int = 80

    @classmethod
    def load(cls, path: str, legacy_path: Optional[str] = None) -> "AppConfig":
        """从 path 读取配置；不存在时尝试 legacy_path，最终回退默认值。"""
        cfg_path = path
        if not os.path.isfile(cfg_path) and legacy_path and os.path.isfile(legacy_path):
            cfg_path = legacy_path
        if not os.path.isfile(cfg_path):
            return cls()
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """从字典构造，忽略类型不符的字段以保证健壮性。"""
        cfg = cls()
        cfg.ffmpeg_path = data.get("ffmpeg_path") or None
        cfg.output_dir = data.get("output_dir") or None
        for field in ("remove_audio", "crop_9x16", "remove_after_export"):
            value = data.get(field)
            if isinstance(value, bool):
                setattr(cfg, field, value)
        for field in ("encoder", "blur_method"):
            value = data.get(field)
            if isinstance(value, str) and value:
                setattr(cfg, field, value)
        strength = data.get("blur_strength")
        if isinstance(strength, int):
            cfg.blur_strength = strength
        return cfg

    def save(self, path: str) -> bool:
        """写入 path，成功返回 True。"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
