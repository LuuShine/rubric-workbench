import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from pydantic import ValidationError

from rubric_tool.llm import generate_benchmark, generate_rubric, test_connection
from rubric_tool.references import ReferenceBundle, extract_reference_files
from rubric_tool.schemas import Rubric
from rubric_tool.validators import ValidationReport, validate_rubric


ROOT = Path(__file__).parent
CSS = (ROOT / "assets" / "app.css").read_text(encoding="utf-8")
CONFIG_PATH = ROOT / ".local" / "api_config.json"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_API_MODEL = "doubao-seed-2-1-turbo-260628"
PUBLIC_DEPLOYMENT = bool(os.getenv("SPACE_ID")) or os.getenv(
    "PUBLIC_DEPLOYMENT", ""
).lower() in {"1", "true", "yes"}


def render_reference_status(bundle: ReferenceBundle) -> str:
    if not bundle.summaries:
        return '<div class="reference-empty">未上传参考文件</div>'

    labels = {"loaded": "已读取", "error": "读取失败", "skipped": "已跳过"}
    items = []
    for item in bundle.summaries:
        items.append(
            f"""
            <div class="reference-item reference-{item.status}">
              <span>{labels[item.status]}</span>
              <strong>{html.escape(item.name)}</strong>
              <small>{html.escape(item.message)}</small>
            </div>
            """
        )
    return f"""
    <div class="reference-status">
      <div class="reference-total">
        已读取 {bundle.loaded_count} 个文件，共 {bundle.total_chars} 字符
      </div>
      {''.join(items)}
    </div>
    """


def handle_reference_upload(files: Any) -> str:
    return render_reference_status(extract_reference_files(files))


def reference_context_for_generation(files: Any) -> str:
    bundle = extract_reference_files(files)
    if bundle.summaries and bundle.loaded_count == 0:
        details = "；".join(
            f"{item.name}：{item.message}" for item in bundle.summaries
        )
        raise ValueError(f"参考文件均未成功读取。{details}")
    return bundle.context


def parse_editor(editor_text: str) -> Rubric:
    if not editor_text.strip():
        raise ValueError("Rubric 编辑器为空。")
    try:
        return Rubric.model_validate(json.loads(editor_text))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 格式错误：第 {error.lineno} 行，第 {error.colno} 列。") from error
    except ValidationError as error:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()[:5]
        )
        raise ValueError(f"Rubric 结构不符合要求：{messages}") from error


def dimension_rows(rubric: Rubric) -> List[List[Any]]:
    return [
        [
            item.id,
            item.name,
            item.weight,
            len(item.positive_examples),
            len(item.negative_examples),
            len(item.scoring_guide),
        ]
        for item in rubric.dimensions
    ]


def render_summary(rubric: Rubric, report: ValidationReport) -> str:
    weight_class = (
        "is-error"
        if report.weight_sum > 1.0001
        else "is-warn"
        if report.weight_sum < 0.9999
        else "is-good"
    )
    return f"""
    <div class="summary-grid">
      <div class="summary-item">
        <span>评分维度</span>
        <strong>{len(rubric.dimensions)}</strong>
      </div>
      <div class="summary-item {weight_class}">
        <span>权重总和</span>
        <strong>{report.weight_sum:.4g}</strong>
      </div>
      <div class="summary-item {'is-error' if report.error_count else 'is-good'}">
        <span>错误</span>
        <strong>{report.error_count}</strong>
      </div>
      <div class="summary-item {'is-warn' if report.warning_count else 'is-good'}">
        <span>警告 / 提醒</span>
        <strong>{report.warning_count + report.info_count}</strong>
      </div>
    </div>
    """


def render_report(report: ValidationReport) -> str:
    relevance_issues = [
        issue for issue in report.issues if issue.code.startswith("example_relevance_")
    ]
    other_issues = [
        issue for issue in report.issues if not issue.code.startswith("example_relevance_")
    ]

    summary_html = f"""
    <div class="relevance-summary">
      <div><span>已检查样例</span><strong>{report.checked_example_count}</strong></div>
      <div><span>低字面重合</span><strong>{report.low_literal_count}</strong></div>
      <div class="keep"><span>建议保留</span><strong>{report.relevance_keep_count}</strong></div>
      <div class="rewrite"><span>建议改写</span><strong>{report.relevance_rewrite_count}</strong></div>
      <div class="delete"><span>建议删除</span><strong>{report.relevance_delete_count}</strong></div>
    </div>
    """

    labels = {"error": "错误", "warning": "警告", "info": "提醒", "pass": "通过"}
    structural_html = []
    for issue in other_issues:
        dimensions = ", ".join(issue.dimensions) or "全局"
        evidence = (
            f"<span>{html.escape(issue.evidence)}</span>" if issue.evidence else ""
        )
        structural_html.append(
            f"""
            <div class="issue issue-{issue.severity}">
              <span class="issue-level">{labels[issue.severity]}</span>
              <div class="issue-copy">
                <strong>{html.escape(issue.message)}</strong>
                {evidence}
              </div>
              <span class="issue-dimensions">{html.escape(dimensions)}</span>
            </div>
            """
        )

    action_titles = {
        "delete": "建议删除",
        "rewrite": "建议场景化改写",
        "keep": "建议保留",
    }
    relevance_sections = []
    for action in ("delete", "rewrite", "keep"):
        action_issues = [issue for issue in relevance_issues if issue.action == action]
        if not action_issues:
            continue
        cards = []
        for issue in action_issues:
            cards.append(
                f"""
                <div class="relevance-card relevance-{action}">
                  <div class="relevance-card-head">
                    <strong>{html.escape(issue.dimension_name)}</strong>
                    <span>{html.escape(issue.example_type)}样例 {issue.example_index}</span>
                  </div>
                  <p>{html.escape(issue.evidence)}</p>
                  <small>{html.escape(issue.suggestion)}</small>
                </div>
                """
            )
        relevance_sections.append(
            f"""
            <section class="relevance-group">
              <h4>{action_titles[action]} <span>{len(action_issues)}</span></h4>
              {''.join(cards)}
            </section>
            """
        )

    if not report.issues:
        return f"""
        <div class="report">
          {summary_html}
          <div class="report-empty">本地校验通过，未发现需要处理的问题。</div>
        </div>
        """

    structural_section = ""
    if structural_html:
        structural_section = f"""
        <section class="structural-issues">
          <h4>结构与一致性问题 <span>{len(structural_html)}</span></h4>
          {''.join(structural_html)}
        </section>
        """

    return f"""
    <div class="report">
      {summary_html}
      <p class="report-note">
        相关性采用任务关键词、领域词、维度词和动作词的本地启发式判断；
        “低字面重合”不等于语义错误。
      </p>
      {structural_section}
      {''.join(relevance_sections)}
    </div>
    """


def validate_and_format(
    rubric: Rubric,
    task_description: str,
    model_type: str,
    evaluation_goal: str,
) -> Tuple[List[List[Any]], str, Dict[str, Any], str]:
    report = validate_rubric(rubric, task_description, model_type, evaluation_goal)
    rubric_dict = rubric.model_dump(mode="json")
    return (
        dimension_rows(rubric),
        render_report(report),
        rubric_dict,
        render_summary(rubric, report),
    )


def handle_generate(
    task_description: str,
    model_type: str,
    evaluation_goal: str,
    dimension_count: int,
    score_min: int,
    score_max: int,
    output_language: str,
    reference_files: Any,
    api_key: str,
    base_url: str,
    api_model: str,
    timeout: float,
    temperature: float,
):
    try:
        reference_context = reference_context_for_generation(reference_files)
        rubric, raw_response = generate_rubric(
            api_key=api_key,
            base_url=base_url,
            api_model=api_model,
            timeout=timeout,
            temperature=temperature,
            task_description=task_description,
            model_type=model_type,
            evaluation_goal=evaluation_goal,
            dimension_count=int(dimension_count),
            score_min=int(score_min),
            score_max=int(score_max),
            output_language=output_language,
            reference_context=reference_context,
        )
        rows, report_html, final_json, summary_html = validate_and_format(
            rubric,
            task_description,
            model_type,
            evaluation_goal,
        )
        editor_text = json.dumps(final_json, ensure_ascii=False, indent=2)
        return (
            editor_text,
            rows,
            report_html,
            raw_response,
            final_json,
            summary_html,
            "",
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def handle_validate(
    editor_text: str,
    task_description: str,
    model_type: str,
    evaluation_goal: str,
):
    try:
        rubric = parse_editor(editor_text)
        return validate_and_format(
            rubric,
            task_description,
            model_type,
            evaluation_goal,
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def handle_export(editor_text: str) -> str:
    try:
        rubric = parse_editor(editor_text)
        payload = rubric.model_dump(mode="json")
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="rubric_",
            encoding="utf-8",
            delete=False,
        ) as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            return output.name
    except Exception as error:
        raise gr.Error(str(error)) from error


def handle_generate_benchmark(
    task_description: str,
    model_type: str,
    evaluation_goal: str,
    domain: str,
    output_type: str,
    risk_level: str,
    reference_files: Any,
    editor_text: str,
    api_key: str,
    base_url: str,
    api_model: str,
    timeout: float,
    temperature: float,
) -> str:
    try:
        rubric = parse_editor(editor_text)
        reference_context = reference_context_for_generation(reference_files)
        return generate_benchmark(
            api_key=api_key,
            base_url=base_url,
            api_model=api_model,
            timeout=timeout,
            temperature=temperature,
            task_description=task_description,
            model_type=model_type,
            evaluation_goal=evaluation_goal,
            domain=domain,
            output_type=output_type,
            risk_level=risk_level,
            rubric=rubric,
            reference_context=reference_context,
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def handle_export_benchmark(markdown: str) -> str:
    if not markdown.strip():
        raise gr.Error("请先生成 Benchmark 设计方案。")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="benchmark_",
        encoding="utf-8",
        delete=False,
    ) as output:
        output.write(markdown)
        return output.name


def connection_status(
    api_key: str,
    base_url: str,
    api_model: str,
    timeout: float,
) -> str:
    result = test_connection(api_key, base_url, api_model, timeout)
    return f'<div class="connection-status">{html.escape(result)}</div>'


def save_local_api_config(api_key: str, base_url: str, api_model: str) -> str:
    if PUBLIC_DEPLOYMENT:
        raise gr.Error("公开部署不保存 API Key。配置仅在当前浏览器会话中使用。")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "api_key": api_key.strip(),
        "base_url": base_url.strip() or DEFAULT_BASE_URL,
        "api_model": api_model.strip() or DEFAULT_API_MODEL,
    }
    descriptor = os.open(
        CONFIG_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, ensure_ascii=False, indent=2)
    return '<div class="storage-status is-saved">已保存到本机配置文件</div>'


def clear_local_api_config():
    if PUBLIC_DEPLOYMENT:
        return (
            "",
            DEFAULT_BASE_URL,
            DEFAULT_API_MODEL,
            '<div class="storage-status">当前会话配置已清除</div>',
        )
    CONFIG_PATH.unlink(missing_ok=True)
    return (
        "",
        DEFAULT_BASE_URL,
        DEFAULT_API_MODEL,
        '<div class="storage-status">本机保存已清除</div>',
    )


def load_local_api_config():
    if PUBLIC_DEPLOYMENT:
        return "", DEFAULT_BASE_URL, DEFAULT_API_MODEL
    if not CONFIG_PATH.exists():
        return "", DEFAULT_BASE_URL, DEFAULT_API_MODEL
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return (
            str(config.get("api_key") or ""),
            str(config.get("base_url") or DEFAULT_BASE_URL),
            str(config.get("api_model") or DEFAULT_API_MODEL),
        )
    except (OSError, json.JSONDecodeError):
        return "", DEFAULT_BASE_URL, DEFAULT_API_MODEL


EMPTY_SUMMARY = """
<div class="summary-grid">
  <div class="summary-item"><span>评分维度</span><strong>0</strong></div>
  <div class="summary-item"><span>权重总和</span><strong>0</strong></div>
  <div class="summary-item"><span>错误</span><strong>0</strong></div>
  <div class="summary-item"><span>警告 / 提醒</span><strong>0</strong></div>
</div>
"""

EMPTY_REPORT = """
<div class="report">
  <div class="report-empty">生成 Rubric 后显示本地校验结果。</div>
</div>
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_padding="0 0 6px",
        block_label_shadow="none",
        block_label_text_color="#17202a",
        block_label_text_size="12px",
        block_label_text_weight="600",
    )

    with gr.Blocks(
        theme=theme,
        css=CSS,
        title="Rubric Workbench",
    ) as demo:
        gr.HTML(
            """
            <header id="app-header">
              <div>
                <h1>Rubric Workbench</h1>
                <p>评测规则生成与本地质量校验</p>
              </div>
            </header>
            """
        )

        with gr.Accordion("API 配置", open=False, elem_id="api-drawer"):
            gr.HTML(
                """
                <p class="sidebar-note">点击“保存到本机”后，配置会在刷新和重启后自动恢复。</p>
                <div class="config-help">
                  <strong>怎么填</strong>
                  <p>Base URL 填 API 根路径，通常到 <code>/v1</code> 或供应商指定版本为止。</p>
                  <p>不要填 <code>/chat/completions</code>；工具会自动调用 Chat Completions。</p>
                </div>
                """
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="sk-... / ark-...",
                elem_id="api-key-input",
            )
            base_url = gr.Textbox(
                label="Base URL",
                value=DEFAULT_BASE_URL,
                placeholder="https://api.example.com/v1",
                elem_id="base-url-input",
            )
            api_model = gr.Textbox(
                label="API 模型名称",
                value=DEFAULT_API_MODEL,
                placeholder="gpt-4.1-mini",
                elem_id="api-model-input",
            )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.2,
                    step=0.1,
                    label="Temperature",
                )
                timeout = gr.Slider(
                    minimum=15,
                    maximum=300,
                    value=120,
                    step=15,
                    label="请求超时（秒）",
                )
            connection_button = gr.Button("测试连接", elem_id="connection-btn")
            connection_result = gr.HTML(
                '<div class="connection-status">尚未测试连接</div>'
            )
            with gr.Column(elem_id="config-storage-footer"):
                with gr.Row(elem_id="config-storage-actions"):
                    save_config_button = gr.Button(
                        "保存到本机",
                        size="sm",
                        elem_id="save-config-btn",
                        visible=not PUBLIC_DEPLOYMENT,
                    )
                    clear_config_button = gr.Button(
                        "清除保存",
                        size="sm",
                        elem_id="clear-config-btn",
                    )
                config_storage_status = gr.HTML(
                    (
                        '<div class="storage-status">API Key 仅在当前会话使用，不会保存</div>'
                        if PUBLIC_DEPLOYMENT
                        else '<div class="storage-status">配置未持久化</div>'
                    ),
                    elem_id="config-storage-status",
                )

        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=1, min_width=0, elem_id="workspace"):
                gr.HTML(
                    '<p class="step-label">01 / 定义评测任务</p>',
                    elem_id="input-step",
                )
                with gr.Group(elem_id="input-panel"):
                    task_description = gr.Textbox(
                        label="评测任务描述",
                        lines=5,
                        placeholder="例如：评估文生图模型生成的电商商品主图质量……",
                    )
                    with gr.Row():
                        model_type = gr.Dropdown(
                            choices=["文本模型", "图像模型", "视频模型", "多模态模型"],
                            value="文本模型",
                            label="评测对象类型",
                        )
                        evaluation_goal = gr.Textbox(
                            label="评估目标",
                            lines=2,
                            placeholder="例如：筛选满足上线标准的模型输出",
                        )
                    domain = gr.Textbox(
                        label="行业 / 领域",
                        placeholder="例如：电商售后、医疗问答、内容安全",
                    )
                    with gr.Row():
                        output_type = gr.Dropdown(
                            choices=["文本", "图片", "视频", "多模态"],
                            value="文本",
                            label="输出形态",
                        )
                        risk_level = gr.Dropdown(
                            choices=["低", "中", "高"],
                            value="中",
                            label="风险等级",
                        )

                    with gr.Accordion("参考文件（可选）", open=False):
                        reference_files = gr.File(
                            label="上传参考文件",
                            file_count="multiple",
                            file_types=[
                                ".txt",
                                ".md",
                                ".json",
                                ".csv",
                                ".pdf",
                                ".docx",
                            ],
                            type="filepath",
                            elem_id="reference-files",
                        )
                        gr.HTML(
                            """
                            <p class="reference-note">
                              支持 TXT、MD、JSON、CSV、PDF、DOCX；最多 6 个，
                              单文件不超过 10 MB。文件仅在当前会话中解析，不会持久化。
                            </p>
                            """
                        )
                        reference_status = gr.HTML(
                            '<div class="reference-empty">未上传参考文件</div>',
                            elem_id="reference-status",
                        )

                    with gr.Accordion("生成约束", open=False):
                        with gr.Row():
                            dimension_count = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                                label="维度数量",
                            )
                            score_min = gr.Number(value=1, precision=0, label="最低分")
                            score_max = gr.Number(value=5, precision=0, label="最高分")
                            output_language = gr.Dropdown(
                                choices=["简体中文", "English"],
                                value="简体中文",
                                label="输出语言",
                            )

                    gr.Examples(
                        examples=[
                            [
                                "评估客服模型对用户退款问题的回答质量。",
                                "文本模型",
                                "判断回复是否正确、完整、礼貌，并能推动问题解决。",
                            ],
                            [
                                "评估文生图模型生成的电商商品主图。",
                                "图像模型",
                                "筛选主体清晰、构图合理且不存在明显视觉错误的图片。",
                            ],
                            [
                                "评估医疗问答模型对常见症状咨询的回复质量。",
                                "文本模型",
                                "检查医学事实、风险提示和就医建议是否准确且不过度诊断。",
                            ],
                            [
                                "评估短视频生成模型的产品宣传片成片质量。",
                                "视频模型",
                                "筛选画面连续、主体稳定、节奏合理且符合品牌要求的视频。",
                            ],
                            [
                                "评估金融研报信息抽取模型的结构化结果。",
                                "文本模型",
                                "检查关键指标提取是否完整、数值是否准确且来源可追溯。",
                            ],
                            [
                                "评估多模态内容审核模型对图文违规风险的识别能力。",
                                "多模态模型",
                                "验证风险分类准确性、误报率和判断依据的可解释性。",
                            ],
                            [
                                "评估教育模型生成的小学数学解题讲解。",
                                "文本模型",
                                "判断推理步骤是否正确、表达是否适龄并能有效启发学生。",
                            ],
                            [
                                "评估界面截图理解模型输出的产品体验分析。",
                                "多模态模型",
                                "检查其是否准确识别交互问题并给出具体、可执行的改进建议。",
                            ],
                        ],
                        inputs=[task_description, model_type, evaluation_goal],
                        label="示例任务",
                        elem_id="task-examples",
                    )

                gr.HTML(
                    '<p class="step-label">02 / 生成与校验结果</p>',
                    elem_id="output-step",
                )
                with gr.Group(elem_id="action-panel"):
                    with gr.Row(elem_id="primary-actions"):
                        generate_button = gr.Button(
                            "生成 Rubric",
                            variant="primary",
                            elem_id="generate-btn",
                        )
                        benchmark_button = gr.Button(
                            "生成 Benchmark",
                            variant="secondary",
                            elem_id="benchmark-btn",
                        )
                    with gr.Row(elem_id="secondary-actions"):
                        validate_button = gr.Button(
                            "运行本地校验",
                            variant="secondary",
                            elem_id="validate-btn",
                        )
                        export_button = gr.DownloadButton(
                            "导出 JSON",
                            variant="secondary",
                            elem_id="export-btn",
                        )
                        benchmark_export_button = gr.DownloadButton(
                            "导出 MD",
                            variant="secondary",
                            elem_id="benchmark-export-btn",
                        )
                summary = gr.HTML(EMPTY_SUMMARY, elem_id="summary-panel")

                with gr.Tabs(elem_id="output-tabs"):
                    with gr.Tab("Rubric 编辑器"):
                        gr.HTML(
                            '<p id="editor-note">编辑后点击“运行本地校验”更新结果。</p>'
                        )
                        rubric_editor = gr.Code(
                            value="{}",
                            language="json",
                            label="Rubric JSON",
                            lines=24,
                            interactive=True,
                        )
                    with gr.Tab("维度概览"):
                        dimension_table = gr.Dataframe(
                            headers=[
                                "ID",
                                "评分维度",
                                "权重",
                                "正向样例",
                                "负向样例",
                                "评分档位",
                            ],
                            datatype=["str", "str", "number", "number", "number", "number"],
                            value=[],
                            interactive=False,
                            label="维度概览",
                        )
                    with gr.Tab("校验报告"):
                        validation_report = gr.HTML(EMPTY_REPORT)
                    with gr.Tab("Benchmark 方案"):
                        benchmark_markdown = gr.Markdown(
                            value="生成并校验 Rubric 后，点击“生成 Benchmark”。",
                            elem_id="benchmark-output",
                        )
                    with gr.Tab("AI 原始响应"):
                        raw_response = gr.Code(
                            value="",
                            language="json",
                            label="未经处理的模型响应",
                            lines=22,
                            interactive=False,
                        )
                    with gr.Tab("最终 JSON"):
                        final_json = gr.JSON(value={}, label="结构化结果")

        connection_button.click(
            fn=connection_status,
            inputs=[api_key, base_url, api_model, timeout],
            outputs=connection_result,
        )

        save_config_button.click(
            fn=save_local_api_config,
            inputs=[api_key, base_url, api_model],
            outputs=config_storage_status,
            queue=False,
            show_api=False,
        )

        clear_config_button.click(
            fn=clear_local_api_config,
            inputs=None,
            outputs=[api_key, base_url, api_model, config_storage_status],
            queue=False,
            show_api=False,
        )

        reference_files.change(
            fn=handle_reference_upload,
            inputs=reference_files,
            outputs=reference_status,
            queue=False,
            show_api=False,
        )

        generate_button.click(
            fn=handle_generate,
            inputs=[
                task_description,
                model_type,
                evaluation_goal,
                dimension_count,
                score_min,
                score_max,
                output_language,
                reference_files,
                api_key,
                base_url,
                api_model,
                timeout,
                temperature,
            ],
            outputs=[
                rubric_editor,
                dimension_table,
                validation_report,
                raw_response,
                final_json,
                summary,
                benchmark_markdown,
            ],
        )

        benchmark_button.click(
            fn=handle_generate_benchmark,
            inputs=[
                task_description,
                model_type,
                evaluation_goal,
                domain,
                output_type,
                risk_level,
                reference_files,
                rubric_editor,
                api_key,
                base_url,
                api_model,
                timeout,
                temperature,
            ],
            outputs=benchmark_markdown,
        )

        validate_button.click(
            fn=handle_validate,
            inputs=[rubric_editor, task_description, model_type, evaluation_goal],
            outputs=[dimension_table, validation_report, final_json, summary],
        )

        export_button.click(
            fn=handle_export,
            inputs=rubric_editor,
            outputs=export_button,
        )

        benchmark_export_button.click(
            fn=handle_export_benchmark,
            inputs=benchmark_markdown,
            outputs=benchmark_export_button,
        )

        demo.load(
            fn=load_local_api_config,
            inputs=None,
            outputs=[api_key, base_url, api_model],
            queue=False,
            show_api=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=4).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
        show_error=True,
    )
