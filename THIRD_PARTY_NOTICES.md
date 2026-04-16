# Third-Party Notices

This document identifies third-party open-source software that this project depends on, interoperates with, or was conceptually informed by. It is provided for attribution and license compliance purposes.

This notice file does not transfer any rights or obligations under the licenses of the listed projects. Each dependency's license governs the terms under which that dependency may be used, copied, modified, or distributed. Users who redistribute this project or its derivatives, or who bundle any of the listed dependencies, are responsible for complying with each applicable upstream license.

---

## Project License

This repository is licensed under the **Apache License 2.0**.

Full license text: [LICENSE](LICENSE)

Copyright notice: Copyright (c) 2024–2026 the contributors of this repository.

---

## Why Apache License 2.0

Apache License 2.0 was selected because:

- It is compatible with the permissive licenses of the primary upstream dependencies (Apache 2.0, MIT, BSD-3-Clause).
- It includes an explicit patent grant, which provides additional protection for users and contributors.
- It permits use, modification, and redistribution in both open-source and commercial contexts, subject to the attribution and notice requirements in Section 4 of the license.
- It does not impose copyleft obligations on application-layer code that merely imports or calls these libraries at runtime.

---

## Runtime Dependencies

The following packages are imported and used at runtime. Their source code is not bundled in this repository. They are installed separately by the user via the provided `requirements-*.txt` files.

---

### Unsloth

| Field | Value |
|-------|-------|
| Repository | https://github.com/unslothai/unsloth |
| PyPI package | `unsloth` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2023 unslothai and contributors |

**Usage in this project:**
- `FastLanguageModel.from_pretrained()` for CUDA model loading with optional 4-bit quantization
- `FastLanguageModel.get_peft_model()` for LoRA adapter preparation
- Inference and export integration via the loaded model object

**Notes:**
The Unsloth repository contains multiple components and subprojects. This project uses only the public Python API exposed by the `unsloth` PyPI package. If you later vendor any source code from non-Apache subdirectories of the Unsloth repository, you must re-examine the license of those specific files before doing so.

---

### MLX-Tune

| Field | Value |
|-------|-------|
| Repository | https://github.com/ARahim3/mlx-tune |
| PyPI package | `mlx-tune` |
| Observed license | Apache License 2.0 |

**Usage in this project:**
- `FastLanguageModel` for Apple Silicon model loading
- `SFTTrainer` / `SFTConfig` for supervised fine-tuning on MLX
- `DPOTrainer` / `DPOConfig` for direct preference optimization on MLX
- `ORPOTrainer` / `ORPOConfig` for odds ratio preference optimization on MLX

---

### MLX

| Field | Value |
|-------|-------|
| Repository | https://github.com/ml-explore/mlx |
| PyPI package | `mlx` |
| Observed license | MIT License |
| Copyright | Copyright (c) 2023 Apple Inc. |

**Usage in this project:**
- `mlx.core` for Apple Silicon native tensor operations
- `mlx.core.clear_cache()` (or `mlx.core.metal.clear_cache()` on older versions) for Metal GPU memory management between training sessions

---

### MLX-LM

| Field | Value |
|-------|-------|
| Repository | https://github.com/ml-explore/mlx-lm |
| PyPI package | `mlx-lm` |
| Observed license | MIT License |
| Copyright | Copyright (c) 2023 Apple Inc. |

**Usage in this project:**
- `mlx_lm.tuner.callbacks.TrainingCallback` for injecting training metrics into the MLX training loop via monkey-patching

---

### Hugging Face Transformers

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/transformers |
| PyPI package | `transformers` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2018 The Hugging Face team |

**Usage in this project:**
- `AutoModelForCausalLM` and `AutoTokenizer` for model and tokenizer loading on HF backends
- `TrainingArguments` for configuring the HF training loop
- `TrainerCallback` for injecting real-time metrics into the training loop
- `BitsAndBytesConfig` for 4-bit quantization on CUDA

---

### Hugging Face Datasets

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/datasets |
| PyPI package | `datasets` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2020 The Hugging Face team |

**Usage in this project:**
- `Dataset.from_list()` for converting Python lists of formatted records into HuggingFace Dataset objects for use with TRL trainers

---

### Hugging Face PEFT

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/peft |
| PyPI package | `peft` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2022 The Hugging Face team |

**Usage in this project:**
- `LoraConfig` for defining LoRA adapter configuration
- `get_peft_model()` for wrapping a base model with LoRA adapters on HF backends
- `adapter_config.json` format for checkpoint compatibility checking

---

### Hugging Face TRL

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/trl |
| PyPI package | `trl` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2022 The Hugging Face team |

**Usage in this project:**
- `SFTTrainer` and `SFTConfig` for supervised fine-tuning
- `DPOTrainer` and `DPOConfig` for direct preference optimization
- `ORPOTrainer` and `ORPOConfig` for odds ratio preference optimization

---

### Hugging Face Accelerate

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/accelerate |
| PyPI package | `accelerate` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2021 The Hugging Face team |

**Usage in this project:**
- Indirect dependency required by Transformers and TRL for distributed training utilities and device placement

---

### Hugging Face Hub

| Field | Value |
|-------|-------|
| Repository | https://github.com/huggingface/huggingface_hub |
| PyPI package | `huggingface-hub` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2021 The Hugging Face team |

**Usage in this project:**
- Model weight download from HuggingFace Hub during training initialization
- `push_to_hub()` integration in the export tab for uploading fine-tuned models

---

### Gradio

| Field | Value |
|-------|-------|
| Repository | https://github.com/gradio-app/gradio |
| PyPI package | `gradio` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2021 Gradio |

**Usage in this project:**
- Complete web UI framework: all components, layouts, event system, and SSE streaming
- `gr.Blocks`, `gr.Tab`, `gr.Row`, `gr.Column`, `gr.Accordion` for layout
- Input/output components: `gr.Textbox`, `gr.Slider`, `gr.Dropdown`, `gr.Radio`, `gr.Checkbox`, `gr.CheckboxGroup`, `gr.File`, `gr.Dataframe`, `gr.HTML`, `gr.Markdown`, `gr.LinePlot`, `gr.BarPlot`, `gr.State`
- `gr.Request` for per-session identification and localhost detection
- Generator-mode event handlers for real-time streaming updates

**Notes:**
This project does not use Gradio's built-in `gr.I18n` system. All multilingual label switching is implemented via pure Python `gr.update()` callbacks with a custom translation dictionary in `ui/i18n.py`.

---

### PyTorch

| Field | Value |
|-------|-------|
| Repository | https://github.com/pytorch/pytorch |
| PyPI package | `torch` |
| Observed license | BSD-3-Clause (with additional component licenses) |
| Copyright | Copyright (c) 2016 Facebook Inc. (now Meta Platforms) |

**Usage in this project:**
- Model execution on CUDA and MPS backends
- `torch.cuda.is_available()` and `torch.backends.mps.is_available()` for hardware detection
- `torch.cuda.memory_allocated()` for GPU memory reporting
- `torch.cuda.empty_cache()` for CUDA memory release after training sessions
- `torch.bfloat16` / `torch.float16` / `torch.float32` dtype selection

**Notes:**
PyTorch bundles several third-party components with their own licenses (including NVIDIA CUDA libraries, Intel MKL, and others). See the [PyTorch license file](https://github.com/pytorch/pytorch/blob/main/LICENSE) and the [PyTorch third-party notices](https://github.com/pytorch/pytorch/blob/main/NOTICE) for the complete list.

---

### bitsandbytes

| Field | Value |
|-------|-------|
| Repository | https://github.com/bitsandbytes-foundation/bitsandbytes |
| PyPI package | `bitsandbytes` |
| Observed license | MIT License |
| Copyright | Copyright (c) 2021 Tim Dettmers |

**Usage in this project:**
- `BitsAndBytesConfig` (via Transformers) for 4-bit NF4 quantization on CUDA
- 8-bit AdamW optimizer (`adamw_8bit`) for memory-efficient training on CUDA

---

### Optuna

| Field | Value |
|-------|-------|
| Repository | https://github.com/optuna/optuna |
| PyPI package | `optuna` |
| Observed license | MIT License |
| Copyright | Copyright (c) 2018 Preferred Networks, Inc. |

**Usage in this project:**
- `optuna.create_study()` with TPE sampler for Bayesian hyperparameter optimization
- `optuna.pruners.MedianPruner` for early stopping of unpromising trials
- `FanovaImportanceEvaluator` for parameter importance analysis
- This dependency is optional; the Auto-Tune tab is disabled if Optuna is not installed

---

### Plotly

| Field | Value |
|-------|-------|
| Repository | https://github.com/plotly/plotly.py |
| PyPI package | `plotly` |
| Observed license | MIT License |
| Copyright | Copyright (c) 2016-2018 Plotly, Inc. |

**Usage in this project:**
- Indirect dependency used by Gradio's `gr.LinePlot` and `gr.BarPlot` components for rendering loss curves, learning rate curves, and parameter importance charts

---

### SentencePiece

| Field | Value |
|-------|-------|
| Repository | https://github.com/google/sentencepiece |
| PyPI package | `sentencepiece` |
| Observed license | Apache License 2.0 |
| Copyright | Copyright (c) 2018 Google LLC |

**Usage in this project:**
- Indirect dependency required by some tokenizers (e.g., LLaMA, Mistral) loaded via Hugging Face Transformers

---

### Protocol Buffers (protobuf)

| Field | Value |
|-------|-------|
| Repository | https://github.com/protocolbuffers/protobuf |
| PyPI package | `protobuf` |
| Observed license | BSD-3-Clause |
| Copyright | Copyright (c) 2008 Google Inc. |

**Usage in this project:**
- Indirect dependency required by Hugging Face libraries for model serialization

---

### pandas

| Field | Value |
|-------|-------|
| Repository | https://github.com/pandas-dev/pandas |
| PyPI package | `pandas` |
| Observed license | BSD-3-Clause |
| Copyright | Copyright (c) 2008–2011 AQR Capital Management, LLC; 2011–2024 Open source contributors |

**Usage in this project:**
- `pd.DataFrame` construction for loss curve and learning rate curve data passed to Gradio plot components

---

## Conceptual and Design References

### karpathy/autoresearch

| Field | Value |
|-------|-------|
| Repository | https://github.com/karpathy/autoresearch |

The auto-tuning module's experiment-loop framing — running multiple short probe training runs and using the results to guide subsequent hyperparameter selection — was conceptually influenced by the autonomous research loop described in this project.

**Important clarification:**
- No source code from `karpathy/autoresearch` is copied, bundled, or derived in this repository.
- This reference is cited solely as conceptual inspiration for the design framing.
- The actual implementation uses Optuna's TPE sampler and is entirely original code.
- If source code from that repository is later incorporated, this notice must be updated to reflect the applicable license terms.

---

## Scope of This Notice

This notice covers:
- Packages imported at runtime by this project's Python source code
- Packages that are indirect but essential dependencies of the above
- Projects that conceptually influenced the design

This notice does **not** cover:
- System-level dependencies (operating system, CUDA drivers, Metal runtime) — these are governed by their respective vendor licenses
- Model weights downloaded at runtime — model weights are subject to their own individual licenses (e.g., Llama community license, Qwen license, Mistral license). It is the user's responsibility to review and comply with the license of any model they download and fine-tune using this tool
- User-provided datasets — the user is solely responsible for ensuring they have the right to use any dataset loaded into this tool

---

## No Upstream Copyright Transfer

All copyrights in upstream dependencies remain with their original copyright holders as identified above.

This repository claims copyright only over:
- The original Python source code written specifically for this project
- The UI structure, layout, and component wiring
- The training orchestration logic, session management system, and checkpoint utilities
- The translation dictionaries and multilingual UI system
- The documentation and configuration files

This project does not claim ownership of, and does not transfer any rights in, any upstream library, model architecture, or dataset.

---

## Redistribution Notice

If you redistribute this project or a derivative work:

1. You must retain this `THIRD_PARTY_NOTICES.md` file and the `LICENSE` file.
2. You must comply with the attribution and notice requirements of each upstream dependency's license, particularly Apache License 2.0 Section 4 (which requires preservation of copyright notices, patent notices, disclaimers, and attribution notices).
3. If you bundle any upstream source code directly (rather than importing it as a package), you must include the applicable upstream license text alongside that code.
4. Model weights are not covered by this project's license. Redistribution of fine-tuned model weights requires compliance with the base model's license.
