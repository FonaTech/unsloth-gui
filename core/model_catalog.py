"""
core/model_catalog.py
模型注册表：加载 configs/model_catalog.json，提供查询接口。
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict

_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "model_catalog.json"
)


@dataclass
class ModelEntry:
    family: str
    display_name: str
    hf_id: str                      # 4bit 优化版 HF ID（Unsloth 专用）
    hf_id_full: str                  # 完整精度版 HF ID
    params_b: float
    context_length: int
    min_vram_4bit_gb: float
    min_vram_fp16_gb: float
    default_target_modules: List[str]

    def vram_requirement(self, use_4bit: bool) -> float:
        return self.min_vram_4bit_gb if use_4bit else self.min_vram_fp16_gb

    def is_compatible(self, available_vram_gb: Optional[float], use_4bit: bool) -> bool:
        if available_vram_gb is None:
            return True  # 未知显存，不做限制
        return available_vram_gb >= self.vram_requirement(use_4bit)


def _load_catalog() -> List[ModelEntry]:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ModelEntry(
            family=m["family"],
            display_name=m["display_name"],
            hf_id=m["hf_id"],
            hf_id_full=m["hf_id_full"],
            params_b=m["params_b"],
            context_length=m["context_length"],
            min_vram_4bit_gb=m["min_vram_4bit_gb"],
            min_vram_fp16_gb=m["min_vram_fp16_gb"],
            default_target_modules=m["default_target_modules"],
        )
        for m in data["models"]
    ]


# 模块级单例
_CATALOG: Optional[List[ModelEntry]] = None


def get_catalog() -> List[ModelEntry]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _load_catalog()
    return _CATALOG


def get_families() -> List[str]:
    """返回所有模型系列名称（去重、保序）。"""
    seen = set()
    families = []
    for m in get_catalog():
        if m.family not in seen:
            seen.add(m.family)
            families.append(m.family)
    return families


def get_models_by_family(family: str) -> List[ModelEntry]:
    return [m for m in get_catalog() if m.family == family]


def find_by_display_name(name: str) -> Optional[ModelEntry]:
    for m in get_catalog():
        if m.display_name == name:
            return m
    return None


def find_by_hf_id(hf_id: str) -> Optional[ModelEntry]:
    """支持精确匹配 4bit ID 或完整精度 ID。"""
    for m in get_catalog():
        if m.hf_id == hf_id or m.hf_id_full == hf_id:
            return m
    return None


def get_all_display_names() -> List[str]:
    return [m.display_name for m in get_catalog()]


def build_family_model_map() -> Dict[str, List[str]]:
    """返回 {family: [display_name, ...]} 字典，用于 Gradio 下拉联动。"""
    result: Dict[str, List[str]] = {}
    for m in get_catalog():
        result.setdefault(m.family, []).append(m.display_name)
    return result
