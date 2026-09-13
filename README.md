# Rubric Workbench

使用 Gradio 构建的评测 Rubric 生成与本地质量校验工具。

## 功能

- 通过 OpenAI 兼容 Chat Completions API 生成结构化 Rubric
- 编辑并预览模型输出
- 本地检查权重总和、重复维度、样例相关性与评分档位
- 样例相关性采用任务词、领域词、维度词和动作词的分层规则
- 将低字面相关性样例分为“建议保留 / 建议场景化改写 / 建议删除”
- 在校验报告中展示命中词、判断依据和建议动作
- 保留未经处理的模型响应
- 导出最终 Rubric 为 JSON
- 基于当前 Rubric 生成包含 17 个章节的 Benchmark 设计方案
- 在页面预览 Benchmark，并导出为 Markdown
- 上传参考文件，为 Rubric 和 Benchmark 提供业务背景、政策与样例上下文
- API 配置可保存到仅本机可读的配置文件

## 生成方法

Rubric 从评测目标反推需要支持的准入、比较和归因决策，并按需覆盖基础可用性、
任务或场景效果、主观质量三个层级。维度要求互斥、可观察、可归因，评分档位提供
明确边界和样例；业务后验指标仅作为评分体系的外部有效性验证，不直接作为单条
输出的评分项。

Benchmark 方案区分稳定回归集、真实分布集和专项挑战集，按照“使用场景 → 类别
→ 子类别 → 关键失败模式”组织样本，并包含难度分层、控制变量、随机性处理、
盲评、锚点集、隐藏集、版本治理和后验验证建议。具体比例与阈值需根据用户资料
或历史数据确定，系统不会编造固定标准。

导入或编辑旧版 Rubric 时，评分说明必须覆盖分数范围内的每个整数档位。例如 1 到
5 分必须同时提供 1、2、3、4、5 五档说明；只有 1/3/5 的旧标准会被拒绝，需要人工
补充 2/4 的可观察边界。系统不会静默复制相邻档位或自动填充缺失说明。

## 参考文件

任务输入区支持同时上传最多 6 个参考文件：

- 格式：TXT、Markdown、JSON、CSV、PDF、DOCX
- 限制：单文件不超过 10 MB，单文件最多提取 8,000 字符，总计最多 20,000 字符
- PDF 仅提取文本层；扫描版 PDF 需要先进行 OCR
- 文件仅在当前 Gradio 会话中解析，不会写入项目配置或数据库

参考内容会同时传给 Rubric 和 Benchmark 生成。Prompt 明确将文件视为不可信参考资料：
只提取相关事实、术语、约束和样例，不执行文件中的指令；与页面输入冲突时，
以用户填写的任务描述和评估目标为准。

## 启动

```bash
cd rubric-workbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

打开 <http://127.0.0.1:7860>。

## 部署到 Render

仓库包含 `render.yaml`，可以通过 Render Blueprint 直接部署为免费 Python Web
Service。公开环境会自动禁用 API Key 持久化，访客填写的 Key 仅在当前会话使用。

1. 将本目录推送到 GitHub 仓库。
2. 在 Render 控制台选择 **New → Blueprint**。
3. 连接 GitHub 仓库并选择 `render.yaml`。
4. 确认实例类型为 **Free**，然后创建服务。
5. 构建完成后使用 Render 提供的 `onrender.com` 地址。

免费实例闲置后会休眠，首次访问需要等待服务恢复，适合作品集和低频演示。

## API 配置

配置面板需要填写 OpenAI 兼容 API 的三个核心字段：

在 Hugging Face Spaces 等公开环境中，API Key 仅在访客当前会话中使用，不会写入
服务器文件；“保存到本机”按钮会自动隐藏。

API 配置默认收起在页面右上角，点击“API 配置”即可展开。点击“保存到本机”后，
配置会写入权限为 `600` 的 `.local/api_config.json`，刷新或重启服务后自动恢复；
点击“清除保存”可立即删除。该文件已被 Git 忽略，不会进入版本控制。

| 字段 | 填写方式 |
| --- | --- |
| API Key | 服务商提供的密钥，通常以 `sk-` 开头，但不同服务商可能不同 |
| Base URL | API 根路径，通常到 `/v1` 为止 |
| API 模型名称 | 服务商文档中的模型 ID，必须完整一致 |

常见写法：

```text
OpenAI:
Base URL = https://api.openai.com/v1
Model    = gpt-4.1-mini

如果服务商文档写 POST https://example.com/v1/chat/completions:
Base URL = https://example.com/v1
Model    = 文档给出的模型名
```

不要把 `Base URL` 写成 `https://example.com/v1/chat/completions`。应用内部会自动调用 `/chat/completions`。

常见错误：

- `401`：API Key 错误、过期，或鉴权格式不匹配
- `403`：账号没有模型权限
- `404`：Base URL 或模型名错误，尤其常见于把 `/chat/completions` 填进 Base URL
- `429`：额度不足或触发限流
- `Connection error`：网络、域名、代理或 VPN 问题

## 测试

```bash
.venv/bin/python -m pytest -q
```

本地校验器位于 `rubric_tool/validators.py`，不依赖 LLM。
