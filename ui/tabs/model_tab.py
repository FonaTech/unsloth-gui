"""
ui/tabs/model_tab.py
Tab 3: 模型选择（目录 / 自定义 ID / 本地路径）与下载
本地路径支持手动输入、常用目录扫描，以及快速选择。
"""

import os
import gradio as gr
from core.environment import EnvironmentInfo
from core.model_catalog import (
    get_families, get_models_by_family, find_by_display_name,
    build_family_model_map,
)
from ui.i18n import tr, ts, get_choices, register_translatable, t


# ── 本地模型目录扫描 ───────────────────────────────────────────────

def _is_model_dir(path: str) -> bool:
    """判断目录是否像一个 HuggingFace 模型目录。"""
    files = set(_safe_listdir(path))
    has_config = "config.json" in files
    has_tokenizer = bool(files & {"tokenizer.json", "tokenizer_config.json", "tokenizer.model"})
    has_gen_config = "generation_config.json" in files
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_adapter = "adapter_config.json" in files

    # HF 模型至少有 config.json + tokenizer 或 generation_config（即使权重未下载）
    if has_config and (has_tokenizer or has_gen_config or has_safetensors):
        return True
    # LoRA adapter 目录
    if has_adapter:
        return True
    return False


def _safe_listdir(path: str) -> list:
    try:
        return os.listdir(path)
    except Exception:
        return []


def scan_model_dirs(extra_paths: list | None = None) -> list[str]:
    """
    扫描常用路径，返回找到的模型目录列表。
    包括：HF cache、~/models、~/Downloads 下一级目录、当前工作目录等。
    """
    found: list[str] = []
    home = os.path.expanduser("~")

    # 优先扫描路径（含递归深度限制）
    scan_roots = [
        os.path.join(home, "models"),
        os.path.join(home, "Downloads", "Fine-Tuning"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "Desktop"),
        os.path.join(home, ".cache", "huggingface", "hub"),
        os.path.join(home, "huggingface"),
        os.path.join(home, ".ollama", "models"),
        os.getcwd(),
    ]
    if extra_paths:
        scan_roots = list(extra_paths) + scan_roots

    seen: set[str] = set()

    def _scan(root: str, depth: int = 0):
        if depth > 3 or not os.path.isdir(root):
            return
        real = os.path.realpath(root)
        if real in seen:
            return
        seen.add(real)

        # HF hub 目录格式：models--org--name/snapshots/hash/ → 检查两层深
        try:
            for entry in os.scandir(root):
                if not entry.is_dir(follow_symlinks=False):
                    continue
                name = entry.name
                # 跳过隐藏目录（除 HF cache 的 .cache/huggingface 以外）
                if name.startswith(".") and ".cache/huggingface" not in root:
                    continue
                if _is_model_dir(entry.path):
                    found.append(entry.path)
                else:
                    # 再往下一层（HF cache 需要两层：models--x/snapshots/hash/）
                    if depth < 2:
                        _scan(entry.path, depth + 1)
        except PermissionError:
            pass

    for r in scan_roots:
        _scan(r, depth=0)

    # 去重保序
    deduped: list[str] = []
    seen_norm: set[str] = set()
    for p in found:
        norm = os.path.normpath(p)
        if norm not in seen_norm:
            seen_norm.add(norm)
            deduped.append(norm)

    return deduped


def build_model_tab(model_state: gr.State, env_info: EnvironmentInfo) -> dict:
    families = get_families()
    family_model_map = build_family_model_map()
    default_family = families[0] if families else ""
    default_models = [m.display_name for m in get_models_by_family(default_family)] if default_family else []
    default_model = default_models[0] if default_models else ""

    with gr.Tab(tr("tab.model"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.model")
        title_md = gr.Markdown(tr("model.title"))
        register_translatable(title_md, label_key="model.title")

        source_radio = gr.Radio(
            choices=get_choices("model_source"),
            value="catalog",
            label=tr("model.source"),
        )
        register_translatable(source_radio, label_key="model.source", choices_key="model_source")

        # ── 内置目录 ────────────────────────────────────────────────
        with gr.Group(visible=True) as catalog_group:
            catalog_hdr = gr.Markdown(tr("model.catalog.title"))
            register_translatable(catalog_hdr, label_key="model.catalog.title")
            with gr.Row():
                family_dd = gr.Dropdown(
                    choices=families,
                    value=default_family,
                    label=tr("model.family"),
                )
                register_translatable(family_dd, label_key="model.family")
                model_dd = gr.Dropdown(
                    choices=default_models,
                    value=default_model,
                    label=tr("model.name"),
                )
                register_translatable(model_dd, label_key="model.name")

        # ── 自定义 HF ID ─────────────────────────────────────────────
        with gr.Group(visible=False) as hf_id_group:
            hf_hdr = gr.Markdown(tr("model.hf.title"))
            register_translatable(hf_hdr, label_key="model.hf.title")
            hf_id_input = gr.Textbox(
                placeholder=ts("model.hf.placeholder"),
                label=tr("model.hf.label"),
            )
            register_translatable(hf_id_input, label_key="model.hf.label")

        # ── 本地路径 ─────────────────────────────────────────────────
        with gr.Group(visible=False) as local_path_group:
            local_hdr = gr.Markdown(tr("model.local.title"))
            register_translatable(local_hdr, label_key="model.local.title")
            with gr.Row():
                local_path_input = gr.Textbox(
                    placeholder=ts("model.local.placeholder"),
                    label=tr("model.local.label"),
                    scale=3,
                )
                register_translatable(local_path_input, label_key="model.local.label")
                scan_btn = gr.Button(tr("model.local.scan"), variant="secondary", scale=1)
                register_translatable(scan_btn, label_key="model.local.scan")

            scan_dropdown = gr.Dropdown(
                choices=[],
                value=None,
                label=tr("model.local.scan_results"),
                interactive=True,
                visible=False,
            )
            register_translatable(scan_dropdown, label_key="model.local.scan_results")
            scan_status = gr.Textbox(
                value="", label="", interactive=False, visible=False,
            )

        # ── 模型信息卡 ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                model_info_html = gr.HTML(value=_model_info_html(
                    find_by_display_name(default_model), env_info
                ))
            with gr.Column(scale=1):
                select_btn = gr.Button(tr("model.select"), variant="primary")
                register_translatable(select_btn, label_key="model.select")
                select_status = gr.Textbox(label="", interactive=False, value="")

        dl_hdr = gr.Markdown(tr("model.download.title"))
        register_translatable(dl_hdr, label_key="model.download.title")
        dl_body = gr.Markdown(tr("model.download.body"))
        register_translatable(dl_body, label_key="model.download.body")

        # ── 事件 ────────────────────────────────────────────────────

        def on_source_change(source):
            return (
                gr.update(visible=(source == "catalog")),
                gr.update(visible=(source == "hf")),
                gr.update(visible=(source == "local")),
            )

        def on_family_change(family):
            models = [m.display_name for m in get_models_by_family(family)]
            value = models[0] if models else None
            return gr.update(choices=models, value=value)

        def on_model_dd_change(display_name):
            entry = find_by_display_name(display_name)
            return _model_info_html(entry, env_info)

        def on_scan(current_path: str):
            """扫描本地常用目录，返回找到的模型路径列表。"""
            extra = [current_path.strip()] if current_path.strip() and os.path.isdir(current_path.strip()) else []
            found = scan_model_dirs(extra_paths=extra)
            if not found:
                return (
                    gr.update(choices=[], visible=True),
                    gr.update(value=t("model.scan.none"), visible=True),
                )
            labels = []
            for p in found:
                name = os.path.basename(p)
                home = os.path.expanduser("~")
                short = p.replace(home, "~")
                labels.append((f"{name}  [{short}]", p))
            return (
                gr.update(choices=labels, value=None, visible=True),
                gr.update(value=t("model.scan.found").format(n=len(found)), visible=True),
            )

        def on_scan_select(selected_path: str):
            """下拉选择后自动填入路径输入框，并展示模型信息。"""
            if not selected_path or not os.path.isdir(selected_path):
                return selected_path or "", "<p>请选择模型</p>"
            # 检查是否为已知内置模型
            html = _local_path_info_html(selected_path)
            return selected_path, html

        def on_select(source, family, model_name, hf_id, local_path):
            if source == "catalog":
                entry = find_by_display_name(model_name)
                if entry is None:
                    return {}, t("model.err.not_found")
                use_4bit = env_info.cuda_available
                model_id = entry.hf_id if use_4bit else entry.hf_id_full
                info = {
                    "model_id": model_id,
                    "display_name": entry.display_name,
                    "load_in_4bit": use_4bit,
                    "default_target_modules": entry.default_target_modules,
                    "context_length": entry.context_length,
                }
                return info, t("model.selected").format(name=entry.display_name, id=model_id)

            elif source == "hf":
                mid = hf_id.strip()
                if not mid:
                    return {}, t("model.err.no_hf_id")
                info = {
                    "model_id": mid,
                    "display_name": mid,
                    "load_in_4bit": env_info.cuda_available,
                    "default_target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    "context_length": 4096,
                }
                return info, t("model.selected_hf").format(id=mid)

            else:  # local
                path = local_path.strip()
                if not path:
                    return {}, t("model.err.no_path")
                if not os.path.isdir(path):
                    return {}, t("model.err.path_not_found").format(path=path)
                info = {
                    "model_id": path,
                    "display_name": os.path.basename(path),
                    "load_in_4bit": False,
                    "default_target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    "context_length": 4096,
                }
                return info, t("model.selected_local").format(path=path)

        source_radio.change(
            fn=on_source_change,
            inputs=[source_radio],
            outputs=[catalog_group, hf_id_group, local_path_group],
        )
        family_dd.change(
            fn=on_family_change,
            inputs=[family_dd],
            outputs=[model_dd],
        )
        model_dd.change(
            fn=on_model_dd_change,
            inputs=[model_dd],
            outputs=[model_info_html],
        )
        scan_btn.click(
            fn=on_scan,
            inputs=[local_path_input],
            outputs=[scan_dropdown, scan_status],
        )
        scan_dropdown.change(
            fn=on_scan_select,
            inputs=[scan_dropdown],
            outputs=[local_path_input, model_info_html],
        )
        select_btn.click(
            fn=on_select,
            inputs=[source_radio, family_dd, model_dd, hf_id_input, local_path_input],
            outputs=[model_state, select_status],
        )

    return {"model_source": source_radio}


# ── HTML helpers ────────────────────────────────────────────────────

def _model_info_html(entry, env_info: EnvironmentInfo) -> str:
    if entry is None:
        return f"<p>{t('model.info.select')}</p>"

    use_4bit = env_info.cuda_available
    vram_req = entry.vram_requirement(use_4bit)
    available = env_info.gpu_vram_gb

    if available is None:
        vram_compat = t("model.info.vram.unknown")
        compat_color = "#fef08a"
    elif available >= vram_req:
        vram_compat = t("model.info.vram.ok").format(avail=available, req=vram_req)
        compat_color = "#bbf7d0"
    else:
        vram_compat = t("model.info.vram.low").format(avail=available, req=vram_req)
        compat_color = "#fecaca"

    precision = t("model.info.precision.4bit") if use_4bit else t("model.info.precision.fp16")

    return f"""
<div style="padding:12px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;font-size:14px;line-height:1.8;">
  <strong style="font-size:16px;">{entry.display_name}</strong><br/>
  <span style="color:#64748b;">{entry.hf_id_full}</span><br/><br/>
  {t('model.info.params')}：<strong>{entry.params_b}B</strong> &nbsp;|&nbsp;
  {t('model.info.context')}：<strong>{entry.context_length:,} tokens</strong><br/>
  {t('model.info.precision')}：<strong>{precision}</strong><br/>
  {t('model.info.vram')}：<strong>{vram_req} GB</strong><br/><br/>
  <div style="padding:6px 10px;border-radius:4px;background:{compat_color};">
    {vram_compat}
  </div>
</div>
"""


def _local_path_info_html(path: str) -> str:
    name = os.path.basename(path)
    config_path = os.path.join(path, "config.json")
    arch_info = ""
    try:
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [t("model.info.no_config")])[0]
        hidden = cfg.get("hidden_size", "?")
        n_layers = cfg.get("num_hidden_layers", "?")
        vocab = cfg.get("vocab_size", "?")
        ctx = cfg.get("max_position_embeddings", "?")
        arch_info = (
            f"{t('model.info.arch')}：<strong>{arch}</strong><br/>"
            f"{t('model.info.hidden')}：{hidden} &nbsp;|&nbsp; "
            f"{t('model.info.layers')}：{n_layers} &nbsp;|&nbsp; "
            f"{t('model.info.vocab')}：{vocab}<br/>"
            f"{t('model.info.max_pos')}：<strong>{ctx}</strong>"
        )
    except Exception:
        arch_info = f"<span style='color:#94a3b8;'>{t('model.info.no_config')}</span>"

    has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    adapter_tag = " &nbsp;<span style='background:#dbeafe;color:#1d4ed8;padding:2px 6px;border-radius:4px;font-size:12px;'>LoRA adapter</span>" if has_adapter else ""

    return f"""
<div style="padding:12px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;font-size:14px;line-height:1.8;">
  <strong style="font-size:16px;">{name}</strong>{adapter_tag}<br/>
  <span style="color:#64748b;font-size:12px;">{path}</span><br/><br/>
  {arch_info}
</div>
"""
