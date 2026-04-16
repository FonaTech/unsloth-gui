# Unsloth GUI Fine-Tuning Workbench

`unsloth-gui` is a local visual fine-tuning tool for LLM adaptation workflows. It wraps common single-machine training flows into one Gradio web UI and can route across:

- `Unsloth + CUDA`
- `MLX-Tune + Apple Silicon`
- `Hugging Face + CUDA / MPS / CPU`

The project focuses on practical local workflows:

- Dataset loading and preview
- Base model selection
- LoRA / SFT / DPO / ORPO configuration
- Live training monitoring
- Export to LoRA / merged model / GGUF
- Optuna-based hyperparameter search

## Status

This repository is under active refinement.

Current state in this version:

- Unified layout and reduced tab width jitter
- Improved visual shell and top-level UI styling
- Initial multilingual foundation for `简体中文 / 繁體中文 / English / 日本語`
- Parameter-facing input components are being migrated to Gradio i18n markers
- Core training / export / tuning logic remains local-first and mostly unchanged

Important note:

- Static UI labels, placeholders, and many parameter hints are now connected to the multilingual layer.
- Some runtime-generated backend messages are still primarily Chinese and should be migrated further if you want fully localized status outputs.

## Requirements

### CUDA / NVIDIA

- Python 3.9+
- PyTorch with CUDA support
- Recommended: `Unsloth`

Install:

```bash
pip install -r requirements-cuda.txt
```

If needed, install an Unsloth build matching your CUDA / torch version.

### macOS / Apple Silicon

- Python 3.9+
- PyTorch with MPS support
- Recommended: `mlx-tune`

Install:

```bash
pip install -r requirements-mps.txt
```

### CPU fallback

Install:

```bash
pip install -r requirements-cpu.txt
```

CPU training is only for debugging or very small experiments.

## Launch

```bash
python app.py
```

Options:

```bash
python app.py --host 0.0.0.0 --port 7860
python app.py --no-browser
python app.py --share
```

## Feature Overview

### 1. Environment Detection

- Detects CUDA / MPS / MLX / CPU
- Shows package versions and install suggestions

### 2. Dataset Setup

- Supports `.jsonl`, `.json`, `.csv`
- Supports SFT-style records and preference records
- Includes split ratio, prompt template, and `<think>` block handling

### 3. Model Selection

- Built-in catalog
- Hugging Face model IDs
- Local model paths
- Local directory scan

### 4. Training Configuration

- LoRA settings
- Scheduler and optimization parameters
- Memory optimization options
- Save / load configuration file

### 5. Training Monitor

- Live status
- Progress bar
- Loss / LR plots
- Logs and checkpoints

### 6. Export and Quick Inference

- Save LoRA adapter
- Save merged model
- Export GGUF
- Push to Hugging Face Hub
- Run quick prompt inference

### 7. Auto Tune

- Optuna TPE search
- Trial history
- Parameter importance
- Apply best parameters back to the main training config

## Multilingual UI

Target languages:

- Simplified Chinese
- Traditional Chinese
- English
- Japanese

Implementation direction:

- Static labels, placeholders, help text, and option labels are being moved to a centralized translation dictionary in [`ui/i18n.py`](/Users/Fona/Downloads/Fine-Tuning/unsloth-gui/ui/i18n.py)
- Shared styling is centralized in [`ui/theme.py`](/Users/Fona/Downloads/Fine-Tuning/unsloth-gui/ui/theme.py)

If you want complete multilingual coverage, continue migrating:

- Markdown section headings
- Runtime validation messages
- Training / export / auto-tune callback outputs
- HTML helper cards in the model tab

## License

This project is licensed under the Apache License 2.0.

See:

- [LICENSE](/Users/Fona/Downloads/Fine-Tuning/unsloth-gui/LICENSE)
- [THIRD_PARTY_NOTICES.md](/Users/Fona/Downloads/Fine-Tuning/unsloth-gui/THIRD_PARTY_NOTICES.md)

## Third-Party Dependencies and References

This project depends on or references several open-source projects, including:

- Unsloth
- MLX-Tune
- Optuna
- Hugging Face Transformers / Datasets / PEFT / TRL / Accelerate
- Gradio
- bitsandbytes
- Plotly
- PyTorch
- SentencePiece

In addition, the auto-tuning workflow and research-loop framing were influenced by:

- `karpathy/autoresearch`

This repository does not bundle those upstream projects' source code wholesale. Their copyrights remain with their respective authors.

See [THIRD_PARTY_NOTICES.md](/Users/Fona/Downloads/Fine-Tuning/unsloth-gui/THIRD_PARTY_NOTICES.md) for attribution and license notes.

## Known Gaps

- Full per-request runtime localization is not complete yet
- Some UI sections still contain hard-coded Chinese Markdown and status strings
- Browser-language selection is currently more reliable than the in-page language dropdown for complete Gradio-side translation changes
- No automated test suite is included yet

## Suggested Next Steps

- Finish translating runtime callback messages
- Move tab titles and Markdown section headers to i18n keys
- Add screenshots / GIFs to the README
- Add smoke tests for UI startup and key backend paths

