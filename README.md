# Unsloth GUI Fine-Tuning Workbench

`unsloth-gui` is a local, single-machine visual fine-tuning workbench for large language models. It provides a Gradio-based web interface that unifies dataset preparation, model selection, LoRA configuration, training monitoring, checkpoint management, model export, and Bayesian hyperparameter search into one cohesive workflow.

The project is designed for researchers and practitioners who want to run fine-tuning experiments locally without writing training scripts from scratch. It routes automatically to the best available backend on the current hardware.

---

## Supported Backends

| Backend | Platform | Notes |
|---------|----------|-------|
| Unsloth + CUDA | Linux / Windows (NVIDIA GPU) | Fastest path; requires `unsloth` package |
| MLX-Tune + Apple Silicon | macOS M-series (M1/M2/M3/M4) | Native Metal GPU; requires `mlx-tune` |
| Hugging Face + CUDA | Linux / Windows (NVIDIA GPU) | Fallback when Unsloth is unavailable |
| Hugging Face + MPS | macOS Apple Silicon | PyTorch MPS backend |
| Hugging Face + CPU | Any platform | Debugging and very small experiments only |

Backend selection is automatic at startup based on detected hardware and installed packages.

---

## Features

### Dataset
- Loads `.jsonl`, `.json`, and `.csv` files
- Supports SFT format (`instruction` / `input` / `output` / `system`) and preference format (`prompt` / `chosen` / `rejected`)
- Configurable train/eval split ratio
- Prompt template selection: Alpaca, ChatML, Llama 3
- `<think>` block handling: keep or strip chain-of-thought reasoning blocks
- Dataset statistics, field validation, and formatted prompt preview

### Model Selection
- Built-in catalog of 17+ models (Llama 3, Qwen 2.5, Mistral, Gemma 2, Phi-4, DeepSeek)
- Custom HuggingFace model ID input
- Local model path with automatic directory scan
- VRAM compatibility check against detected GPU memory

### Training Configuration
- LoRA hyperparameters: rank, alpha, dropout, target modules, RSLoRA
- Training type: SFT, DPO, ORPO
- Optimizer, scheduler, learning rate, warmup, weight decay, gradient clipping
- Memory optimization: 4-bit quantization, gradient checkpointing, sequence packing, NEFTune
- Save/load configuration as JSON
- Quick presets: Quick Test, Memory Efficient, Balanced, High Quality

### Training Monitor
- Real-time status, progress bar, ETA
- Live loss and learning rate plots
- GPU memory usage and training speed
- Full training log stream
- Checkpoint list
- Checkpoint resume: scan existing checkpoints, verify config compatibility, resume from any step
- Session reconnect: page refresh reconnects to a running training session without interruption

### Export
- Save LoRA adapter weights
- Save merged full-precision model (fp16)
- Export GGUF with configurable quantization level
- Push to HuggingFace Hub
- Quick inference test against the loaded model

### Auto-Tune
- Optuna TPE Bayesian optimization
- Configurable search space: LoRA rank, learning rate, batch size, warmup ratio, scheduler, gradient accumulation
- Trial history table and convergence plot
- Parameter importance analysis (FAnova)
- Apply best parameters back to the main training configuration

### Multilingual UI
- Languages: Simplified Chinese, Traditional Chinese, English (default), Japanese
- All labels, choices, section headers, status messages, and error strings are translated
- Language preference persisted in `localStorage`; restored on page reload
- Pure Python `gr.update()` switching — no dependency on Gradio's internal i18n system

### Color Themes
- Six built-in themes: Ocean Blue, Forest Green, Amber Sand, Rose Clay, Indigo Mist, Slate Mono
- CSS variable switching; theme preference persisted in `localStorage`

### Multi-User and Session Management
- **Single-user mode** (default): all sessions share one training instance — suitable for personal local use
- **Per-session mode**: each browser tab receives an independent monitor, orchestrator, and auto-tuner instance
- **Task queue**: when running jobs reach the configured maximum, new training requests queue automatically and start when a slot opens
- **GPU release**: on training completion, model references are deleted and GPU/Metal cache is cleared to free memory for the next session
- **Admin panel**: visible only to localhost users; controls mode toggle and max concurrent jobs
- **Session status bar**: shows online sessions, active training count, and queue depth

### Network Access
- Binds to `0.0.0.0` by default — accessible from both `http://localhost:7860` and the local network IP
- Browser opens automatically to `http://localhost:7860`
- Both URLs printed at startup

---

## Requirements

Python 3.9 or later is required for all configurations.

### macOS / Apple Silicon

Recommended for M1/M2/M3/M4 machines. Uses MLX-Tune for native Metal GPU training.

```bash
pip install -r requirements-mps.txt
```

### CUDA / NVIDIA GPU

Recommended for Linux and Windows with NVIDIA hardware. Installs Unsloth for the fastest training path.

```bash
pip install -r requirements-cuda.txt
```

If your CUDA or PyTorch version requires a specific Unsloth build, follow the [Unsloth installation guide](https://github.com/unslothai/unsloth#installation) and install the matching wheel separately.

### CPU Fallback

For debugging, testing, or machines without a supported GPU.

```bash
pip install -r requirements-cpu.txt
```

CPU training is extremely slow for any model larger than a few hundred million parameters. It is not recommended for real fine-tuning runs.

---

## Launch

```bash
python app.py
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port PORT` | `7860` | HTTP server port |
| `--host HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` to restrict to localhost only |
| `--no-browser` | off | Suppress automatic browser launch |
| `--share` | off | Create a temporary public Gradio share URL |

### Example

```bash
# LAN-accessible, no browser
python app.py --no-browser

# Localhost only
python app.py --host 127.0.0.1

# Custom port
python app.py --port 8080
```

---

## Project Structure

```
app.py                          Entry point and CLI argument handling
core/
  trainer.py                    Training orchestration: Unsloth / MLX / HF backends
  monitor.py                    Thread-safe queue-based metrics collection
  dataset.py                    Dataset loading, validation, formatting, statistics
  environment.py                Platform and backend detection
  checkpoint.py                 Checkpoint scanning and LoRA config compatibility check
  session_manager.py            Session registry, isolation modes, task queue, GPU release
  auto_tuner.py                 Optuna TPE hyperparameter search
  exporter.py                   LoRA / merged model / GGUF export, Hub push, inference
  model_catalog.py              Built-in model definitions and VRAM estimates
ui/
  app_builder.py                Gradio application assembly, admin panel, session status
  i18n.py                       Translation dictionaries (4 languages), component registry
  theme.py                      CSS variable themes, localStorage persistence
  tabs/
    env_tab.py                  Tab 1: Environment detection
    dataset_tab.py              Tab 2: Dataset configuration
    model_tab.py                Tab 3: Model selection
    config_tab.py               Tab 4: Training hyperparameters and presets
    training_tab.py             Tab 5: Training monitor, checkpoint resume, queue flow
    export_tab.py               Tab 6: Export and quick inference
    auto_tune_tab.py            Tab 7: Bayesian hyperparameter auto-tuning
configs/
  presets/
    quick_test.json             1 epoch, 100 samples
    memory_efficient.json       Optimized for 8 GB VRAM
    balanced.json               Balanced for 16 GB VRAM
    high_quality.json           High quality for 24 GB+ VRAM
```

---

## Configuration Presets

Presets are stored as JSON files in `configs/presets/` and can be loaded from the Training Configuration tab. You can also save your own configuration from the UI and reload it later.

---

## Checkpoint Resume

The training tab includes a checkpoint resume section. To resume a previous run:

1. Enter the output directory path in the scan field
2. Click **Scan Checkpoints** — the tool detects both HuggingFace-format checkpoint directories (`checkpoint-{step}/`) and MLX adapter files (`{step}_adapters.safetensors`)
3. Select a checkpoint from the dropdown
4. The compatibility checker verifies that the base model, LoRA rank, alpha, and target modules match the current configuration
5. Enable **Resume from checkpoint** and start training

---

## Session and Concurrency Model

By default the application runs in **single-user mode**: all browser sessions share one training process. This is appropriate for personal local use.

To enable multi-user isolation, open the Admin panel (visible only when accessing from `localhost`) and switch to **Per-session mode**. In this mode:

- Each browser tab receives its own `TrainingMonitor`, `TrainingOrchestrator`, and `AutoTuner` instance
- Sessions do not interfere with each other
- The **Max concurrent training jobs** slider controls how many sessions can train simultaneously
- Sessions that exceed the limit are placed in a queue and start automatically when a slot becomes available
- When a session finishes training, its model references are released and GPU/Metal memory is cleared before the next session starts

---

## License

This project is licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for the full license text.

---

## Third-Party Dependencies

This project depends on several open-source libraries. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full list of dependencies, their licenses, and attribution notes.

---

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied. The authors are not responsible for any damages, data loss, or other consequences arising from the use of this software. Fine-tuning large language models involves significant computational resources and may produce outputs that require careful evaluation before deployment.

Model weights downloaded through this tool are subject to their own respective licenses. It is the user's responsibility to comply with the terms of any model license before using, distributing, or deploying fine-tuned models.
