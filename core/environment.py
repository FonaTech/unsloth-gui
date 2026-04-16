"""
core/environment.py
平台与硬件环境检测模块：检测 CUDA / MPS / CPU，确定最佳训练后端。
"""

import sys
import platform
import importlib.util
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class EnvironmentInfo:
    platform_name: str          # "darwin" | "windows" | "linux"
    python_version: str
    backend: str                # "unsloth_cuda" | "hf_cuda" | "mlx" | "hf_mps" | "hf_cpu"
    cuda_available: bool
    cuda_version: Optional[str]
    mps_available: bool
    rocm_available: bool
    mlx_available: bool
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_count: int
    torch_version: str
    unsloth_available: bool
    packages: Dict[str, str]    # pkg_name -> version_string
    warnings: List[str]


def _get_package_version(pkg_name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "未安装"


def detect_environment() -> EnvironmentInfo:
    """检测当前运行环境，返回 EnvironmentInfo 对象。"""
    try:
        import torch
    except ImportError:
        # torch 未安装，返回最小环境信息
        return EnvironmentInfo(
            platform_name=platform.system().lower(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            backend="hf_cpu",
            cuda_available=False,
            cuda_version=None,
            mps_available=False,
            rocm_available=False,
            gpu_name=None,
            gpu_vram_gb=None,
            gpu_count=0,
            torch_version="未安装",
            unsloth_available=False,
            packages={},
            warnings=["PyTorch 未安装，请先安装依赖。"],
        )

    warnings: List[str] = []
    plat = platform.system().lower()  # "darwin" | "windows" | "linux"
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # ── CUDA ──────────────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    cuda_version: Optional[str] = torch.version.cuda if cuda_available else None
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_count = 0

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = round(props.total_memory / (1024 ** 3), 1)

    # ── ROCm (AMD GPU) ────────────────────────────────────────────────────
    rocm_available = (
        cuda_available
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    )

    # ── MPS (Apple Silicon) ───────────────────────────────────────────────
    mps_available = False
    if plat == "darwin":
        try:
            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception:
            pass
        if mps_available and gpu_name is None:
            try:
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.split("\n"):
                    if "Chipset Model" in line:
                        gpu_name = line.split(":", 1)[1].strip()
                        break
                if gpu_name is None:
                    gpu_name = "Apple Silicon GPU"
            except Exception:
                gpu_name = "Apple Silicon GPU"

    # ── Unsloth ───────────────────────────────────────────────────────────
    unsloth_available = importlib.util.find_spec("unsloth") is not None

    # ── MLX-Tune (Apple Silicon native) ──────────────────────────────────
    mlx_available = (
        plat == "darwin"
        and importlib.util.find_spec("mlx_tune") is not None
    )

    # ── Backend selection ─────────────────────────────────────────────────
    if cuda_available and unsloth_available:
        backend = "unsloth_cuda"
    elif cuda_available:
        backend = "hf_cuda"
        warnings.append(
            "检测到 CUDA，但 Unsloth 未安装。将使用标准 HuggingFace 后端（速度较慢）。"
            "\n安装 Unsloth 可显著提升训练速度：pip install unsloth"
        )
    elif mlx_available:
        backend = "mlx"
        # No warning — MLX is the optimal Mac backend
    elif mps_available:
        backend = "hf_mps"
        warnings.append(
            "macOS MPS 模式：不支持 bitsandbytes 4bit 量化。"
            "\n将使用 bfloat16 精度，LoRA 训练仍然有效。"
            "\n安装 mlx-tune 可获得 Apple Silicon 原生加速：pip install mlx-tune"
        )
    else:
        backend = "hf_cpu"
        warnings.append(
            "未检测到 GPU 加速。CPU 训练速度非常慢，建议使用配备 GPU 的机器。"
        )

    # ── Package versions ──────────────────────────────────────────────────
    packages: Dict[str, str] = {}
    for pkg in [
        "torch", "transformers", "trl", "peft",
        "accelerate", "datasets", "gradio",
        "bitsandbytes", "unsloth", "mlx_tune", "mlx", "plotly",
    ]:
        packages[pkg] = _get_package_version(pkg)

    return EnvironmentInfo(
        platform_name=plat,
        python_version=python_version,
        backend=backend,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        mps_available=mps_available,
        rocm_available=rocm_available,
        mlx_available=mlx_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
        torch_version=packages.get("torch", "未安装"),
        unsloth_available=unsloth_available,
        packages=packages,
        warnings=warnings,
    )


def get_install_instructions(env: EnvironmentInfo) -> str:
    """根据当前环境返回对应的安装建议。"""
    if env.backend in ("unsloth_cuda", "mlx"):
        return "✅ 所有依赖已就绪，可以直接开始训练。"

    if env.platform_name == "darwin":
        return (
            "macOS 推荐安装（Apple Silicon 原生加速）：\n"
            "  pip install -r requirements-mps.txt\n"
            "  pip install mlx-tune\n\n"
            "注意：macOS 不支持 bitsandbytes 4bit 量化，训练将使用 bfloat16/fp16 精度。\n"
            "MLX-Tune 可利用 Apple Silicon 统一内存，无需 MPS 兼容层。"
        )
    elif env.cuda_available:
        return (
            f"检测到 CUDA {env.cuda_version}，安装 Unsloth 以获得最佳性能：\n"
            f"  pip install -r requirements-cuda.txt\n\n"
            f"或单独安装 Unsloth：\n"
            f"  pip install unsloth"
        )
    else:
        return (
            "CPU 模式安装命令：\n"
            "  pip install -r requirements-cpu.txt\n\n"
            "⚠️  CPU 训练极慢，强烈建议使用带 GPU 的机器。"
        )


def get_backend_display(backend: str) -> str:
    """返回后端的可读名称。"""
    labels = {
        "unsloth_cuda": "Unsloth + CUDA (最优)",
        "hf_cuda": "HuggingFace + CUDA",
        "mlx": "MLX-Tune + Apple Silicon (原生最优)",
        "hf_mps": "HuggingFace + Apple MPS",
        "hf_cpu": "HuggingFace + CPU (极慢)",
    }
    return labels.get(backend, backend)


def supports_4bit(env: EnvironmentInfo) -> bool:
    """是否支持 4bit 量化（需要 bitsandbytes + CUDA）。"""
    return env.cuda_available and env.packages.get("bitsandbytes", "未安装") != "未安装"
