"""
core/exporter.py
模型导出模块：支持 LoRA 适配器、合并完整模型、GGUF 量化、HuggingFace Hub 推送。
"""

import os
import subprocess
import sys
from typing import Optional, Callable


LogFn = Callable[[str], None]


def _log(fn: Optional[LogFn], msg: str) -> None:
    if fn:
        fn(msg)
    else:
        print(msg)


# ────────────────────────────────────────────────────────────────
# LoRA Adapter
# ────────────────────────────────────────────────────────────────

def save_lora_adapter(
    model,
    tokenizer,
    output_dir: str,
    log_fn: Optional[LogFn] = None,
) -> str:
    """保存 LoRA 适配器（仅增量权重，10-100MB）。"""
    os.makedirs(output_dir, exist_ok=True)
    _log(log_fn, f"保存 LoRA 适配器到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log(log_fn, "LoRA 适配器保存完成。")
    return output_dir


# ────────────────────────────────────────────────────────────────
# Merged Model
# ────────────────────────────────────────────────────────────────

def save_merged_model(
    model,
    tokenizer,
    output_dir: str,
    log_fn: Optional[LogFn] = None,
) -> str:
    """合并 LoRA 权重到基础模型并保存完整模型。"""
    os.makedirs(output_dir, exist_ok=True)
    _log(log_fn, f"合并模型到: {output_dir}（可能需要几分钟...）")

    # 尝试 Unsloth 方式
    try:
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        _log(log_fn, "合并完成（Unsloth merged_16bit）。")
        return output_dir
    except AttributeError:
        pass

    # 标准 PEFT 方式
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _log(log_fn, "合并完成（PEFT merge_and_unload）。")
        return output_dir
    except Exception as e:
        raise RuntimeError(f"合并模型失败: {e}") from e


# ────────────────────────────────────────────────────────────────
# GGUF
# ────────────────────────────────────────────────────────────────

GGUF_QUANTIZATIONS = {
    "q4_k_m": "Q4_K_M（推荐，4bit，质量/大小均衡）",
    "q5_k_m": "Q5_K_M（5bit，质量更高）",
    "q8_0":   "Q8_0（8bit，接近全精度）",
    "f16":    "F16（半精度，最大质量）",
    "q2_k":   "Q2_K（2bit，极小体积，质量较低）",
}


def save_gguf(
    model,
    tokenizer,
    output_dir: str,
    quantization: str = "q4_k_m",
    log_fn: Optional[LogFn] = None,
) -> str:
    """
    将模型导出为 GGUF 格式。
    优先使用 Unsloth 内置方法；若不可用，尝试通过 llama.cpp 转换。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 方式 1：Unsloth 内置 GGUF 导出
    try:
        _log(log_fn, f"导出 GGUF ({quantization}) 到: {output_dir}")
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
        gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
        if gguf_files:
            path = os.path.join(output_dir, gguf_files[0])
            _log(log_fn, f"GGUF 导出完成: {path}")
            return path
    except AttributeError:
        _log(log_fn, "Unsloth GGUF 方法不可用，尝试先合并模型再转换...")
    except Exception as e:
        _log(log_fn, f"Unsloth GGUF 失败: {e}，尝试备用方法...")

    # 方式 2：先合并为完整模型，再调用 llama.cpp 转换
    merged_dir = os.path.join(output_dir, "merged_for_gguf")
    save_merged_model(model, tokenizer, merged_dir, log_fn)

    # 查找 llama.cpp convert 脚本
    convert_script = _find_llama_cpp_convert()
    if convert_script is None:
        raise RuntimeError(
            "未找到 llama.cpp convert 脚本。\n"
            "请先安装 llama.cpp 或使用 Unsloth CUDA 后端（支持内置 GGUF 导出）。"
        )

    gguf_out = os.path.join(output_dir, f"model-{quantization}.gguf")
    cmd = [
        sys.executable, convert_script,
        merged_dir,
        "--outfile", gguf_out,
        "--outtype", quantization,
    ]
    _log(log_fn, f"运行 llama.cpp 转换: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp 转换失败:\n{result.stderr}")

    _log(log_fn, f"GGUF 导出完成: {gguf_out}")
    return gguf_out


def _find_llama_cpp_convert() -> Optional[str]:
    """查找 llama.cpp convert_hf_to_gguf.py 或 convert.py 脚本。"""
    candidates = [
        os.path.join(os.getcwd(), "llama.cpp", "convert_hf_to_gguf.py"),
        os.path.join(os.getcwd(), "llama.cpp", "convert.py"),
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ────────────────────────────────────────────────────────────────
# HuggingFace Hub
# ────────────────────────────────────────────────────────────────

def push_to_hub(
    model,
    tokenizer,
    repo_id: str,
    token: str,
    private: bool = True,
    log_fn: Optional[LogFn] = None,
) -> str:
    """将模型和 tokenizer 推送到 HuggingFace Hub。"""
    _log(log_fn, f"推送到 HuggingFace Hub: {repo_id}")
    try:
        model.push_to_hub(repo_id, token=token, private=private)
        tokenizer.push_to_hub(repo_id, token=token, private=private)
        url = f"https://huggingface.co/{repo_id}"
        _log(log_fn, f"推送完成: {url}")
        return url
    except Exception as e:
        raise RuntimeError(f"推送失败: {e}") from e


# ────────────────────────────────────────────────────────────────
# Quick Inference
# ────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """使用已加载模型进行快速推理（供导出 Tab 测试用）。"""
    import torch

    # 尝试 Unsloth FastInference
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # 只返回新生成的 token
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
