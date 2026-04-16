# Third-Party Notices

This project depends on, interoperates with, or was informed by third-party open-source software. Ownership of those projects remains with their respective authors and maintainers.

## Project License Choice

This repository is licensed under Apache License 2.0.

Why Apache-2.0:

- It is compatible with the project’s current structure as an application layer built around multiple permissive upstream dependencies.
- It provides explicit patent protection language.
- It is consistent with several important upstream dependencies used directly in the code path.

## Direct Dependencies Referenced in Code

### Unsloth

- Project: `https://github.com/unslothai/unsloth`
- Observed license at the repository root: Apache License 2.0
- Usage in this project:
  - CUDA model loading
  - LoRA preparation
  - inference / export integration

Note:

- The repository currently contains additional components and subprojects. If you later vendor any source code from non-Apache subdirectories, you must re-check those subcomponents separately.

### MLX-Tune

- Project: `https://github.com/ARahim3/mlx-tune`
- Observed license: Apache License 2.0
- Usage in this project:
  - Apple Silicon training backend
  - SFT / DPO / ORPO trainer integration

### Optuna

- Project: `https://github.com/optuna/optuna`
- Observed license: MIT License
- Usage in this project:
  - TPE sampler
  - pruner
  - study / importance analysis

### Hugging Face Transformers

- Project: `https://github.com/huggingface/transformers`
- Observed license: Apache License 2.0
- Usage:
  - model loading
  - tokenizer loading
  - training arguments
  - callback integration

### Hugging Face Datasets

- Project: `https://github.com/huggingface/datasets`
- Observed package metadata license: Apache 2.0
- Usage:
  - dataset container conversion

### Hugging Face PEFT

- Project: `https://github.com/huggingface/peft`
- Observed package metadata license: Apache
- Usage:
  - LoRA configuration and wrapping

### Hugging Face TRL

- Project: `https://github.com/huggingface/trl`
- Observed repository license: Apache License 2.0
- Usage:
  - `SFTTrainer`
  - `DPOTrainer`
  - `ORPOTrainer`

### Accelerate

- Project: `https://github.com/huggingface/accelerate`
- Observed package metadata license: Apache

### Gradio

- Project: `https://github.com/gradio-app/gradio`
- Observed repository license: Apache License 2.0
- Usage:
  - complete web UI framework
  - custom i18n mechanism

### PyTorch

- Project: `https://github.com/pytorch/pytorch`
- Observed package metadata license: BSD-3-Clause
- Usage:
  - model execution
  - CUDA detection
  - generation / training backend support

### bitsandbytes

- Project: `https://github.com/bitsandbytes-foundation/bitsandbytes`
- Observed repository license: MIT License
- Usage:
  - 4-bit quantization path
  - 8-bit optimizer path

### Plotly

- Project: `https://github.com/plotly/plotly.py`
- Observed repository license: MIT License
- Usage:
  - chart rendering through Gradio plotting components

### SentencePiece

- Project: `https://github.com/google/sentencepiece`
- Observed repository license: Apache License 2.0

### Protocol Buffers

- Project: `https://developers.google.com/protocol-buffers/`
- Observed package metadata license: BSD-3-Clause

## Conceptual / Design Reference

### karpathy/autoresearch

- Project: `https://github.com/karpathy/autoresearch`
- This repository’s auto-tuning module explicitly references the idea of an autonomous experiment loop in comments and design framing.

Important clarification:

- This project does not claim ownership of `autoresearch`.
- The repository is cited here as conceptual inspiration for the experiment-loop framing, not as a bundled code dependency.
- If source code from that repository is later copied into this project, attribution and license terms must be updated accordingly.

## Attribution Scope

This repository currently appears to:

- import and use upstream libraries
- adapt ideas and workflows from public projects
- implement its own local application-layer logic around those libraries

This notice file is not a substitute for each dependency’s own license text. If you redistribute binaries, packaged artifacts, or vendored third-party source, you should also preserve all upstream notices required by those projects.

## No Upstream Copyright Transfer

All copyrights in upstream dependencies remain with their original copyright holders.

This repository claims copyright only over the original code, glue logic, UI structure, documentation, and modifications created specifically for this project, excluding third-party code and assets owned by others.

