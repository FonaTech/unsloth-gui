"""
ui/tabs/export_tab.py
Tab 6: Model export and quick inference
"""

import os
import gradio as gr
from core.exporter import GGUF_QUANTIZATIONS
from core.session_manager import session_manager
from ui.i18n import tr, ts, register_translatable


def build_export_tab() -> None:
    with gr.Tab(tr("tab.export"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.export")
        title_md = gr.Markdown(tr("export.title"))
        register_translatable(title_md, label_key="export.title")

        # ── 检查点选择 ────────────────────────────────────────────────
        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                choices=[], value=None,
                label=tr("export.checkpoint"),
                scale=3,
            )
            register_translatable(checkpoint_dd, label_key="export.checkpoint")
            refresh_ckpt_btn = gr.Button(tr("export.refresh"), scale=1)
            register_translatable(refresh_ckpt_btn, label_key="export.refresh")

        # ── 导出格式 ──────────────────────────────────────────────────
        fmt_hdr = gr.Markdown(tr("export.format.title"))
        register_translatable(fmt_hdr, label_key="export.format.title")
        with gr.Row():
            export_lora = gr.Checkbox(value=True, label=tr("export.lora"))
            register_translatable(export_lora, label_key="export.lora")
            export_merged = gr.Checkbox(value=False, label=tr("export.merged"))
            register_translatable(export_merged, label_key="export.merged")
            export_gguf = gr.Checkbox(value=False, label=tr("export.gguf"))
            register_translatable(export_gguf, label_key="export.gguf")

        with gr.Row(visible=True) as gguf_options:
            gguf_quant_dd = gr.Dropdown(
                choices=list(GGUF_QUANTIZATIONS.keys()),
                value="q4_k_m",
                label=tr("export.gguf_quant"),
            )
            register_translatable(gguf_quant_dd, label_key="export.gguf_quant")

        # ── 输出目录 ──────────────────────────────────────────────────
        with gr.Row():
            export_output_dir = gr.Textbox(
                value="exported",
                label=tr("export.output_dir"),
                scale=3,
            )
            register_translatable(export_output_dir, label_key="export.output_dir")
            export_btn = gr.Button(tr("export.start"), variant="primary", scale=1)
            register_translatable(export_btn, label_key="export.start")

        export_log = gr.Textbox(
            value="", label=tr("export.log"), lines=8, interactive=False,
        )
        register_translatable(export_log, label_key="export.log")

        # ── HuggingFace Hub 推送 ──────────────────────────────────────
        with gr.Accordion(tr("export.accordion.hub"), open=False) as acc_hub:
            register_translatable(acc_hub, label_key="export.accordion.hub")
            with gr.Row():
                hub_repo_id = gr.Textbox(
                    placeholder=ts("export.hub_repo.placeholder"),
                    label=tr("export.hub_repo"),
                    scale=2,
                )
                register_translatable(hub_repo_id, label_key="export.hub_repo")
                hub_token = gr.Textbox(
                    placeholder=ts("export.hub_token.placeholder"),
                    label=tr("export.hub_token"),
                    type="password",
                    scale=2,
                )
                register_translatable(hub_token, label_key="export.hub_token")
                hub_private = gr.Checkbox(value=True, label=tr("export.hub_private"), scale=1)
                register_translatable(hub_private, label_key="export.hub_private")
            hub_push_btn = gr.Button(tr("export.hub_push"), variant="secondary")
            register_translatable(hub_push_btn, label_key="export.hub_push")
            hub_status = gr.Textbox(value="", label=tr("export.hub_status"), interactive=False)
            register_translatable(hub_status, label_key="export.hub_status")

        # ── 快速推理测试 ──────────────────────────────────────────────
        infer_hdr = gr.Markdown(tr("export.infer.title"))
        register_translatable(infer_hdr, label_key="export.infer.title")
        infer_note = gr.Markdown(tr("export.infer.note"))
        register_translatable(infer_note, label_key="export.infer.note")
        with gr.Row():
            infer_input = gr.Textbox(
                placeholder=ts("export.infer_input.placeholder"),
                label=tr("export.infer_input"),
                lines=3,
                scale=3,
            )
            register_translatable(infer_input, label_key="export.infer_input")
            with gr.Column(scale=1):
                infer_max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label=tr("export.infer_max_tokens"),
                )
                register_translatable(infer_max_tokens, label_key="export.infer_max_tokens")
                infer_temp = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                    label=tr("export.infer_temp"),
                )
                register_translatable(infer_temp, label_key="export.infer_temp")
        infer_btn = gr.Button(tr("export.infer_run"), variant="secondary")
        register_translatable(infer_btn, label_key="export.infer_run")
        infer_output = gr.Textbox(
            value="", label=tr("export.infer_output"), lines=8, interactive=False,
        )
        register_translatable(infer_output, label_key="export.infer_output")

        # ── Events ──────────────────────────────────────────────────

        export_gguf.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[export_gguf],
            outputs=[gguf_options],
        )

        def refresh_checkpoints(request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            monitor = session_manager.get_or_create(session_id).monitor
            ckpts = monitor.get_checkpoints()
            if not ckpts:
                return gr.update(choices=[], value=None)
            return gr.update(choices=ckpts, value=ckpts[-1])

        refresh_ckpt_btn.click(fn=refresh_checkpoints, inputs=[], outputs=[checkpoint_dd])

        def do_export(ckpt_path, do_lora, do_merged, do_gguf, gguf_quant, out_subdir,
                      request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            sess = session_manager.get_or_create(session_id)
            orchestrator = sess.orchestrator
            monitor = sess.monitor

            if orchestrator._model is None:
                return "❌ Model not loaded. Please complete training first."

            if not ckpt_path:
                ckpts = monitor.get_checkpoints()
                if not ckpts:
                    return "❌ No checkpoints found. Please complete training first."
                ckpt_path = ckpts[-1]

            model = orchestrator._model
            tokenizer = orchestrator._tokenizer
            logs = []
            base_dir = os.path.join(ckpt_path, out_subdir or "exported")

            try:
                if do_lora:
                    from core.exporter import save_lora_adapter
                    out = save_lora_adapter(model, tokenizer, os.path.join(base_dir, "lora"), logs.append)
                    logs.append(f"✅ LoRA adapter: {out}")
                if do_merged:
                    from core.exporter import save_merged_model
                    out = save_merged_model(model, tokenizer, os.path.join(base_dir, "merged"), logs.append)
                    logs.append(f"✅ Merged model: {out}")
                if do_gguf:
                    from core.exporter import save_gguf
                    out = save_gguf(model, tokenizer, os.path.join(base_dir, "gguf"), gguf_quant, logs.append)
                    logs.append(f"✅ GGUF: {out}")
            except Exception as e:
                logs.append(f"❌ Export failed: {e}")

            return "\n".join(logs)

        export_btn.click(
            fn=do_export,
            inputs=[checkpoint_dd, export_lora, export_merged,
                    export_gguf, gguf_quant_dd, export_output_dir],
            outputs=[export_log],
        )

        def do_hub_push(repo_id, token, private, request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            orchestrator = session_manager.get_or_create(session_id).orchestrator
            if orchestrator._model is None:
                return "❌ Model not loaded"
            if not repo_id or not token:
                return "❌ Please fill in Repo ID and Token"
            from core.exporter import push_to_hub
            logs = []
            try:
                url = push_to_hub(orchestrator._model, orchestrator._tokenizer,
                                  repo_id, token, private, logs.append)
                return "\n".join(logs) + f"\n✅ {url}"
            except Exception as e:
                return f"❌ {e}"

        hub_push_btn.click(fn=do_hub_push, inputs=[hub_repo_id, hub_token, hub_private], outputs=[hub_status])

        def do_inference(prompt, max_tokens, temp, request: gr.Request):
            session_id = getattr(request, "session_hash", "__singleton__")
            orchestrator = session_manager.get_or_create(session_id).orchestrator
            if orchestrator._model is None:
                return "❌ Model not loaded. Please complete training first."
            if not prompt.strip():
                return "❌ Please enter a prompt"
            from core.exporter import run_inference
            try:
                return run_inference(orchestrator._model, orchestrator._tokenizer,
                                     prompt.strip(), int(max_tokens), float(temp))
            except Exception as e:
                return f"❌ Inference failed: {e}"

        infer_btn.click(
            fn=do_inference,
            inputs=[infer_input, infer_max_tokens, infer_temp],
            outputs=[infer_output],
        )
