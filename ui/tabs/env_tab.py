"""
ui/tabs/env_tab.py
Tab 1: Environment detection
"""

import gradio as gr
from core.environment import EnvironmentInfo, get_install_instructions, get_backend_display
from ui.i18n import tr, register_translatable


def build_env_tab(env_info: EnvironmentInfo) -> None:
    with gr.Tab(tr("tab.env"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.env")
        title_md = gr.Markdown(tr("env.title"))
        register_translatable(title_md, label_key="env.title")

        with gr.Row():
            with gr.Column(scale=1):
                sys_hdr = gr.Markdown(tr("env.sys_info"))
                register_translatable(sys_hdr, label_key="env.sys_info")
                sys_info = gr.Textbox(
                    label=tr("env.sys_info_label"),
                    value=_format_sys_info(env_info),
                    lines=4,
                    interactive=False,
                )
                register_translatable(sys_info, label_key="env.sys_info_label")

            with gr.Column(scale=1):
                gpu_hdr = gr.Markdown(tr("env.gpu_info"))
                register_translatable(gpu_hdr, label_key="env.gpu_info")
                gpu_info = gr.Textbox(
                    label=tr("env.gpu_info_label"),
                    value=_format_gpu_info(env_info),
                    lines=4,
                    interactive=False,
                )
                register_translatable(gpu_info, label_key="env.gpu_info_label")

        with gr.Row():
            with gr.Column(scale=2):
                backend_hdr = gr.Markdown(tr("env.backend"))
                register_translatable(backend_hdr, label_key="env.backend")
                backend_html = gr.HTML(value=_format_backend_html(env_info))

            with gr.Column(scale=1):
                refresh_btn = gr.Button(tr("env.refresh"), variant="secondary")
                register_translatable(refresh_btn, label_key="env.refresh")

        pkg_hdr = gr.Markdown(tr("env.packages"))
        register_translatable(pkg_hdr, label_key="env.packages")
        packages_df = gr.Dataframe(
            value=_format_packages(env_info),
            headers=["Package", "Version", "Status"],
            interactive=False,
            wrap=True,
        )

        if env_info.warnings:
            warn_hdr = gr.Markdown(tr("env.warnings"))
            register_translatable(warn_hdr, label_key="env.warnings")
            for w in env_info.warnings:
                gr.Markdown(f"> ⚠️  {w}")

            install_box = gr.Textbox(
                label=tr("env.install_cmd"),
                value=get_install_instructions(env_info),
                lines=6,
                interactive=False,
                buttons=["copy"],
            )
            register_translatable(install_box, label_key="env.install_cmd")

        def do_refresh():
            from core.environment import detect_environment
            new_env = detect_environment()
            return (
                _format_sys_info(new_env),
                _format_gpu_info(new_env),
                _format_backend_html(new_env),
                _format_packages(new_env),
            )

        refresh_btn.click(
            fn=do_refresh,
            inputs=[],
            outputs=[sys_info, gpu_info, backend_html, packages_df],
        )


def _format_sys_info(env: EnvironmentInfo) -> str:
    lines = [
        f"OS: {env.platform_name}",
        f"Python: {env.python_version}",
        f"PyTorch: {env.torch_version}",
    ]
    if env.cuda_version:
        lines.append(f"CUDA: {env.cuda_version}")
    return "\n".join(lines)


def _format_gpu_info(env: EnvironmentInfo) -> str:
    if env.gpu_name:
        lines = [f"Name: {env.gpu_name}"]
        if env.gpu_vram_gb:
            lines.append(f"VRAM: {env.gpu_vram_gb} GB")
        if env.gpu_count > 1:
            lines.append(f"GPU count: {env.gpu_count}")
        if env.cuda_available:
            try:
                import torch
                used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                lines.append(f"Used: {used:.1f} GB")
            except Exception:
                pass
        return "\n".join(lines)
    elif env.mps_available:
        return "Apple Silicon MPS\n(shared system memory)"
    else:
        return "No GPU detected\nRunning on CPU"


def _format_backend_html(env: EnvironmentInfo) -> str:
    color_map = {
        "unsloth_cuda": ("#22c55e", "✅"),
        "hf_cuda":      ("#86efac", "✅"),
        "hf_mps":       ("#facc15", "⚡"),
        "hf_cpu":       ("#f87171", "⚠️"),
    }
    color, icon = color_map.get(env.backend, ("#94a3b8", "❓"))
    label = get_backend_display(env.backend)
    return (
        f'<div style="padding:12px;border-radius:8px;background:{color}20;'
        f'border:2px solid {color};font-size:16px;">'
        f'{icon} <strong>Backend: {label}</strong>'
        f'</div>'
    )


def _format_packages(env: EnvironmentInfo) -> list:
    critical = {"torch", "transformers", "trl", "peft", "gradio"}
    rows = []
    for pkg, ver in env.packages.items():
        if ver == "未安装":
            status = "❌ Missing" if pkg in critical else "— Optional"
        else:
            status = "✅"
        rows.append([pkg, ver, status])
    return rows
