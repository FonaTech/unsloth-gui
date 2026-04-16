"""
core/checkpoint.py
Checkpoint scanning, config matching, and resume support.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CheckpointInfo:
    path: str
    step: int
    timestamp: float  # file mtime
    format: str       # "hf" or "mlx"

    @property
    def label(self) -> str:
        name = os.path.basename(self.path)
        return f"Step {self.step} — {name}"


def scan_checkpoints(output_dir: str) -> List[CheckpointInfo]:
    """Scan output_dir for existing training checkpoints (HF and MLX formats)."""
    results: List[CheckpointInfo] = []
    if not output_dir or not os.path.isdir(output_dir):
        return results

    # HF format: output_dir/checkpoint-{step}/ with trainer_state.json or adapter_config.json
    for entry in _safe_scandir(output_dir):
        if not entry.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", entry.name)
        if m:
            step = int(m.group(1))
            # Verify it has actual checkpoint content
            has_state = os.path.exists(os.path.join(entry.path, "trainer_state.json"))
            has_adapter = os.path.exists(os.path.join(entry.path, "adapter_config.json"))
            has_model = any(
                f.endswith(".safetensors") or f.endswith(".bin")
                for f in _safe_listdir(entry.path)
            )
            if has_state or has_adapter or has_model:
                results.append(CheckpointInfo(
                    path=entry.path,
                    step=step,
                    timestamp=entry.stat().st_mtime,
                    format="hf",
                ))

    # MLX format: output_dir/adapters/{step}_adapters.safetensors
    adapters_dir = os.path.join(output_dir, "adapters")
    if os.path.isdir(adapters_dir):
        for entry in _safe_scandir(adapters_dir):
            if not entry.is_file():
                continue
            m = re.match(r"(\d+)_adapters\.safetensors$", entry.name)
            if m:
                step = int(m.group(1))
                results.append(CheckpointInfo(
                    path=entry.path,
                    step=step,
                    timestamp=entry.stat().st_mtime,
                    format="mlx",
                ))

    # Also check for "final" directory
    final_dir = os.path.join(output_dir, "final")
    if os.path.isdir(final_dir):
        has_adapter = os.path.exists(os.path.join(final_dir, "adapter_config.json"))
        has_model = any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in _safe_listdir(final_dir)
        )
        if has_adapter or has_model:
            mtime = os.path.getmtime(final_dir)
            results.append(CheckpointInfo(
                path=final_dir,
                step=999999,
                timestamp=mtime,
                format="hf",
            ))

    results.sort(key=lambda c: c.step)
    return results


def load_checkpoint_config(ckpt_path: str) -> Optional[dict]:
    """
    Read LoRA config from a checkpoint.

    Search order:
    1. adapter_config.json in the checkpoint dir (HF PEFT format)
    2. adapter_config.json in the parent dir (MLX single-file checkpoints)
    3. training_config.json in the checkpoint dir (auto-saved by this app)
    4. training_config.json in the parent dir
    5. training_config.json in the grandparent dir (per-session subdir layout)

    Returns a dict with at least the keys used by configs_compatible():
      r, lora_alpha, target_modules, base_model_name_or_path
    The dict also carries a "_source" key ("adapter_config" or "training_config")
    so callers can show the user where the config came from.
    """
    search_dirs = []

    if os.path.isdir(ckpt_path):
        search_dirs.append(ckpt_path)
        search_dirs.append(os.path.dirname(ckpt_path))
        search_dirs.append(os.path.dirname(os.path.dirname(ckpt_path)))
    elif os.path.isfile(ckpt_path):
        parent = os.path.dirname(ckpt_path)
        search_dirs.append(parent)
        search_dirs.append(os.path.dirname(parent))
        search_dirs.append(os.path.dirname(os.path.dirname(parent)))

    # 1 & 2: adapter_config.json (authoritative PEFT format)
    for d in search_dirs:
        p = os.path.join(d, "adapter_config.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["_source"] = "adapter_config"
                return cfg
            except Exception:
                pass

    # 3, 4, 5: training_config.json (saved by this app alongside every checkpoint)
    for d in search_dirs:
        p = os.path.join(d, "training_config.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    tc = json.load(f)
                # Normalise to adapter_config field names so configs_compatible() works unchanged
                cfg = {
                    "r": tc.get("lora_r"),
                    "lora_alpha": tc.get("lora_alpha"),
                    "target_modules": tc.get("target_modules"),
                    "base_model_name_or_path": tc.get("model_id", ""),
                    "_source": "training_config",
                }
                return cfg
            except Exception:
                pass

    return None


def configs_compatible(ckpt_config: dict, model_id: str, lora_r: int,
                       lora_alpha: int, target_modules: list) -> Tuple[bool, str]:
    """
    Check if a checkpoint's adapter config is compatible with current training config.
    Returns (is_compatible, reason_if_not).
    """
    reasons = []

    # Check base model
    ckpt_model = ckpt_config.get("base_model_name_or_path", "")
    if ckpt_model and ckpt_model != model_id:
        reasons.append(f"Base model mismatch: checkpoint={ckpt_model}, current={model_id}")

    # Check LoRA rank
    ckpt_r = ckpt_config.get("r")
    if ckpt_r is not None and int(ckpt_r) != int(lora_r):
        reasons.append(f"LoRA rank mismatch: checkpoint r={ckpt_r}, current r={lora_r}")

    # Check LoRA alpha
    ckpt_alpha = ckpt_config.get("lora_alpha")
    if ckpt_alpha is not None and int(ckpt_alpha) != int(lora_alpha):
        reasons.append(f"LoRA alpha mismatch: checkpoint={ckpt_alpha}, current={lora_alpha}")

    # Check target modules
    ckpt_modules = ckpt_config.get("target_modules")
    if ckpt_modules is not None:
        ckpt_set = set(ckpt_modules) if isinstance(ckpt_modules, list) else set()
        current_set = set(target_modules) if target_modules else set()
        if ckpt_set and current_set and ckpt_set != current_set:
            reasons.append(f"Target modules mismatch: checkpoint={sorted(ckpt_set)}, current={sorted(current_set)}")

    if reasons:
        return False, "; ".join(reasons)
    return True, "Compatible"


def _safe_scandir(path: str):
    try:
        return list(os.scandir(path))
    except (PermissionError, OSError):
        return []


def _safe_listdir(path: str) -> list:
    try:
        return os.listdir(path)
    except Exception:
        return []
