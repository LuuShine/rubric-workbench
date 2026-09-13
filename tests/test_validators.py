from rubric_tool.schemas import Rubric
from rubric_tool.validators import (
    text_similarity,
    validate_duplicate_dimensions,
    validate_examples,
    validate_rubric,
    validate_weights,
)


def make_rubric(weights=(0.5, 0.5), duplicate=False) -> Rubric:
    second_name = "答案正确性" if duplicate else "表达清晰性"
    return Rubric.model_validate(
        {
            "task_summary": "评估客服模型回答退款问题的质量",
            "score_scale": {"min": 1, "max": 5},
            "dimensions": [
                {
                    "id": "D01",
                    "name": "答案正确性",
                    "description": "回复中的退款政策和步骤必须正确。",
                    "weight": weights[0],
                    "positive_examples": ["准确说明退款条件和办理步骤。"],
                    "negative_examples": ["提供了错误的退款期限。"],
                    "scoring_guide": {
                        "1": "关键信息完全错误",
                        "2": "关键信息多处错误",
                        "3": "主要结论正确但存在遗漏",
                        "4": "主要结论正确且只有轻微瑕疵",
                        "5": "信息完整且准确",
                    },
                },
                {
                    "id": "D02",
                    "name": second_name,
                    "description": (
                        "回复中的退款政策和步骤必须正确。"
                        if duplicate
                        else "回复应当结构清楚且容易理解。"
                    ),
                    "weight": weights[1],
                    "positive_examples": ["分步骤说明退款处理方式。"],
                    "negative_examples": ["表达混乱，用户无法理解。"],
                    "scoring_guide": {
                        "1": "难以理解",
                        "2": "表达混乱且需要大量推断",
                        "3": "基本清楚",
                        "4": "表达清楚但结构略有欠缺",
                        "5": "表达清晰且结构完整",
                    },
                },
            ],
        }
    )


def test_exact_weight_sum_passes() -> None:
    issues = validate_weights(make_rubric())
    assert issues == []


def test_weight_overflow_is_error() -> None:
    issues = validate_weights(make_rubric((0.7, 0.5)))
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].code == "weight_overflow"


def test_weight_underflow_is_warning() -> None:
    issues = validate_weights(make_rubric((0.4, 0.4)))
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].code == "weight_underflow"


def test_duplicate_dimension_is_reported() -> None:
    rubric = make_rubric(duplicate=True)
    issues = validate_duplicate_dimensions(rubric.dimensions)
    assert any(issue.code == "duplicate_dimension" for issue in issues)


def test_synonym_normalization_improves_similarity() -> None:
    assert text_similarity("答案准确性", "答案正确性") == 1.0


def test_full_validation_returns_summary_counts() -> None:
    rubric = make_rubric((0.7, 0.5), duplicate=True)
    report = validate_rubric(
        rubric,
        task_description="评估客服模型对退款问题的回答",
        model_type="文本模型",
        evaluation_goal="确保退款政策正确且表达清晰",
    )
    assert report.weight_sum == 1.2
    assert report.error_count >= 1
    assert report.warning_count >= 1


def test_refund_domain_terms_prevent_false_irrelevance_warning() -> None:
    rubric = Rubric.model_validate(
        {
            "task_summary": "评估客服回答质量",
            "score_scale": {"min": 1, "max": 5},
            "dimensions": [
                {
                    "id": "D01",
                    "name": "问题解决推动力",
                    "description": "能否提供下一步操作",
                    "weight": 1.0,
                    "positive_examples": [
                        "在订单页点击申请退款并上传商品照片，等待审核。"
                    ],
                    "negative_examples": ["只给出结论，没有任何后续操作。"],
                    "scoring_guide": {
                        "1": "无法推进",
                        "2": "只提供模糊方向",
                        "3": "提供部分步骤",
                        "4": "步骤明确但缺少少量条件",
                        "5": "可直接执行",
                    },
                }
            ],
        }
    )
    issues = validate_examples(
        rubric,
        task_description="评估客服文本模型针对用户退款问题的回答质量",
        model_type="文本模型",
        evaluation_goal="有效推动问题解决",
    )

    assert not any(
        issue.code == "example_relevance_rewrite"
        and issue.example_type == "正向"
        for issue in issues
    )


def test_generic_courtesy_example_is_suggested_for_scenario_rewrite() -> None:
    rubric = Rubric.model_validate(
        {
            "task_summary": "评估退款客服回答",
            "score_scale": {"min": 1, "max": 5},
            "dimensions": [
                {
                    "id": "D01",
                    "name": "礼貌与专业性",
                    "description": "语气礼貌、亲切且专业",
                    "weight": 1.0,
                    "positive_examples": [
                        "开头使用您好，结尾表达如有其他问题随时联系我们。"
                    ],
                    "negative_examples": ["使用你自己看规则啊等不耐烦表述。"],
                    "scoring_guide": {
                        "1": "态度恶劣",
                        "2": "语气生硬",
                        "3": "基本礼貌",
                        "4": "礼貌且比较自然",
                        "5": "礼貌专业",
                    },
                }
            ],
        }
    )
    issues = validate_examples(
        rubric,
        task_description="评估客服文本模型针对用户退款问题的回答质量",
        model_type="文本模型",
        evaluation_goal="判断回复是否礼貌并推动退款问题解决",
    )

    relevance = [
        issue for issue in issues if issue.code.startswith("example_relevance_")
    ]
    assert relevance
    assert all(issue.action == "rewrite" for issue in relevance)
    assert any("您好" in issue.dimension_term_hits for issue in relevance)


def test_clearly_unrelated_example_is_suggested_for_deletion() -> None:
    rubric = Rubric.model_validate(
        {
            "task_summary": "评估退款客服回答",
            "score_scale": {"min": 1, "max": 5},
            "dimensions": [
                {
                    "id": "D01",
                    "name": "问题解决推动力",
                    "description": "是否给出可执行步骤",
                    "weight": 1.0,
                    "positive_examples": ["进入个人资料页修改头像并保存。"],
                    "negative_examples": ["没有解释退款申请步骤。"],
                    "scoring_guide": {
                        "1": "无法推进",
                        "2": "只给出含糊建议",
                        "3": "给出部分步骤",
                        "4": "步骤基本完整",
                        "5": "可以解决",
                    },
                }
            ],
        }
    )
    issues = validate_examples(
        rubric,
        task_description="评估客服文本模型针对用户退款问题的回答质量",
        model_type="文本模型",
        evaluation_goal="有效推动退款问题解决",
    )

    delete_issues = [issue for issue in issues if issue.action == "delete"]
    assert len(delete_issues) == 1
    assert "个人资料页" in delete_issues[0].suggestion
