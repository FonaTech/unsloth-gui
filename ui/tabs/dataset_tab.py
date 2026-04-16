"""
ui/tabs/dataset_tab.py
Tab 2: 数据集上传、验证、预览与配置
"""

import os
import gradio as gr
from typing import Optional

from ui.i18n import tr, ts, get_choices, register_translatable, t


def build_dataset_tab(dataset_state: gr.State) -> dict:
    """
    构建数据集 Tab。
    dataset_state: gr.State，保存当前选择的数据集信息字典。
    返回需要动态更新的组件字典。
    """
    with gr.Tab(tr("tab.dataset"), elem_classes="workspace-tab", render_children=True) as tab:
        register_translatable(tab, label_key="tab.dataset")
        title_md = gr.Markdown(tr("dataset.title"))
        register_translatable(title_md, label_key="dataset.title")

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label=tr("dataset.file.label"),
                    file_types=[".jsonl", ".json", ".csv"],
                )
                register_translatable(file_input, label_key="dataset.file.label")
                file_path_box = gr.Textbox(
                    label=tr("dataset.file.path"),
                    placeholder=ts("dataset.file.path.placeholder"),
                    interactive=True,
                )
                register_translatable(file_path_box, label_key="dataset.file.path")
                load_btn = gr.Button(tr("dataset.load"), variant="primary")
                register_translatable(load_btn, label_key="dataset.load")

            with gr.Column(scale=1):
                stats_box = gr.Textbox(
                    label=tr("dataset.stats"),
                    lines=8,
                    interactive=False,
                    value=ts("dataset.stats.empty"),
                )
                register_translatable(stats_box, label_key="dataset.stats")

        # 字段映射
        with gr.Accordion(tr("dataset.accordion.fields"), open=True) as acc_fields:
            register_translatable(acc_fields, label_key="dataset.accordion.fields")
            with gr.Row():
                training_type_select = gr.Radio(
                    choices=get_choices("training_type"),
                    value="sft",
                    label=tr("dataset.training_type"),
                )
                register_translatable(training_type_select, label_key="dataset.training_type", choices_key="training_type")
                template_select = gr.Dropdown(
                    choices=get_choices("template"),
                    value="alpaca",
                    label=tr("dataset.template"),
                )
                register_translatable(template_select, label_key="dataset.template", choices_key="template")
                think_mode_select = gr.Radio(
                    choices=get_choices("think_mode"),
                    value="keep",
                    label=tr("dataset.think"),
                )
                register_translatable(think_mode_select, label_key="dataset.think", choices_key="think_mode")

            train_ratio_slider = gr.Slider(
                minimum=0.8, maximum=0.99, value=0.95, step=0.01,
                label=tr("dataset.train_ratio"),
            )
            register_translatable(train_ratio_slider, label_key="dataset.train_ratio")
            max_samples_input = gr.Number(
                value=0, label=tr("dataset.max_samples"), precision=0,
            )
            register_translatable(max_samples_input, label_key="dataset.max_samples")

        # 验证状态
        validate_status = gr.Textbox(
            label=tr("dataset.validate"), interactive=False, value=""
        )
        register_translatable(validate_status, label_key="dataset.validate")

        # 数据预览
        with gr.Accordion(tr("dataset.accordion.preview_raw"), open=False) as acc_preview:
            register_translatable(acc_preview, label_key="dataset.accordion.preview_raw")
            preview_table = gr.Dataframe(
                headers=["instruction", "input", "output"],
                label=tr("dataset.preview.raw"),
                interactive=False,
                wrap=True,
                row_count=5,
            )
            register_translatable(preview_table, label_key="dataset.preview.raw")

        with gr.Accordion(tr("dataset.accordion.preview_prompt"), open=False) as acc_prompt:
            register_translatable(acc_prompt, label_key="dataset.accordion.preview_prompt")
            prompt_preview = gr.Textbox(
                label=tr("dataset.preview.prompt"),
                lines=15,
                interactive=False,
            )
            register_translatable(prompt_preview, label_key="dataset.preview.prompt")

        apply_btn = gr.Button(tr("dataset.apply"), variant="primary")
        register_translatable(apply_btn, label_key="dataset.apply")
        apply_status = gr.Textbox(label="", interactive=False, value="")

        # ── 事件处理 ──────────────────────────────────────────────

        def on_load(file_obj, manual_path, training_type):
            from core.dataset import (
                load_raw, validate_fields, compute_statistics,
                preview_samples, detect_fields,
            )

            # 确定路径
            if file_obj is not None:
                path = file_obj.name
            elif manual_path and manual_path.strip():
                path = manual_path.strip()
            else:
                return t("dataset.err.no_file"), "", [], "", ""

            if not os.path.isfile(path):
                return t("dataset.err.not_found").format(path=path), "", [], "", ""

            try:
                records = load_raw(path)
            except Exception as e:
                return t("dataset.err.load_fail").format(e=e), "", [], "", ""

            # 验证
            result = validate_fields(records, training_type)
            validate_msg = result.message

            # 统计
            train_recs, eval_recs = records[:int(len(records) * 0.95)], records[int(len(records) * 0.95):]
            stats = compute_statistics(records, len(train_recs), len(eval_recs))
            stats_text = (
                f"{t('dataset.stats.total')}: {stats.total}\n"
                f"{t('dataset.stats.train')}: {stats.train_size} / {t('dataset.stats.eval')}: {stats.eval_size}\n"
                f"{t('dataset.stats.fields')}: {', '.join(stats.fields)}\n"
                f"{t('dataset.stats.avg_instruction')}: {stats.avg_instruction_len:.0f}\n"
                f"{t('dataset.stats.avg_output')}: {stats.avg_output_len:.0f}\n"
                f"{t('dataset.stats.has_think')}: {'✅' if stats.has_think_blocks else '❌'}\n"
                f"{t('dataset.stats.with_input')}: {stats.sample_count_with_input} / {min(500, stats.total)}"
            )

            # 预览
            samples = preview_samples(records, 5)
            fields = detect_fields(records)
            if training_type == "sft":
                rows = [[s.get("instruction", ""), s.get("input", ""), s.get("output", "")] for s in samples]
                headers = ["instruction", "input", "output"]
            else:
                rows = [[s.get("prompt", ""), s.get("chosen", ""), s.get("rejected", "")] for s in samples]
                headers = ["prompt", "chosen", "rejected"]

            return stats_text, validate_msg, rows, path, ""

        def on_template_change(file_obj, manual_path, template, think_mode):
            """更新格式化预览。"""
            from core.dataset import load_raw, format_preview_prompt

            path = None
            if file_obj is not None:
                path = file_obj.name
            elif manual_path and manual_path.strip():
                path = manual_path.strip()

            if not path or not os.path.isfile(path):
                return ""

            try:
                records = load_raw(path)
                if not records:
                    return ""
                return format_preview_prompt(records[0], template)
            except Exception as e:
                return f"预览失败: {e}"

        def on_apply(file_obj, manual_path, training_type, template, think_mode,
                     train_ratio, max_samples):
            path = None
            if file_obj is not None:
                path = file_obj.name
            elif manual_path and manual_path.strip():
                path = manual_path.strip()

            if not path or not os.path.isfile(path):
                return {}, t("dataset.err.no_valid_file")

            info = {
                "path": path,
                "training_type": training_type,
                "template": template,
                "think_mode": think_mode,
                "train_ratio": float(train_ratio),
                "max_samples": int(max_samples),
            }
            return info, t("dataset.applied").format(name=os.path.basename(path), type=training_type.upper())

        # 加载按钮
        load_btn.click(
            fn=on_load,
            inputs=[file_input, file_path_box, training_type_select],
            outputs=[stats_box, validate_status, preview_table, file_path_box, apply_status],
        )

        # 模板改变时更新预览
        template_select.change(
            fn=on_template_change,
            inputs=[file_input, file_path_box, template_select, think_mode_select],
            outputs=[prompt_preview],
        )
        think_mode_select.change(
            fn=on_template_change,
            inputs=[file_input, file_path_box, template_select, think_mode_select],
            outputs=[prompt_preview],
        )

        # 确认按钮
        apply_btn.click(
            fn=on_apply,
            inputs=[file_input, file_path_box, training_type_select, template_select,
                    think_mode_select, train_ratio_slider, max_samples_input],
            outputs=[dataset_state, apply_status],
        )

    return {
        "training_type": training_type_select,
        "template": template_select,
        "think_mode": think_mode_select,
    }
