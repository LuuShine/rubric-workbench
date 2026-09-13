import pytest
from pydantic import ValidationError

from rubric_tool.schemas import Rubric


def valid_payload() -> dict:
    return {
        "task_summary": "评估客服回答",
        "score_scale": {"min": 1, "max": 5},
        "dimensions": [
            {
                "id": "D01",
                "name": "准确性",
                "description": "核对回答是否符合任务事实和约束。",
                "weight": 1.0,
                "positive_examples": ["事实和步骤都正确。"],
                "negative_examples": ["编造不存在的政策。"],
                "scoring_guide": {
                    "1": "核心结论错误",
                    "2": "多处关键信息错误",
                    "3": "主要结论正确但遗漏明显",
                    "4": "结论正确且只有轻微遗漏",
                    "5": "完全准确且可执行",
                },
            }
        ],
    }


def test_requires_complete_integer_score_guide() -> None:
    payload = valid_payload()
    payload["dimensions"][0]["scoring_guide"] = {
        "1": "核心结论错误",
        "3": "主要结论正确但遗漏明显",
        "5": "完全准确且可执行",
    }

    with pytest.raises(ValidationError, match="missing 2, 4"):
        Rubric.model_validate(payload)


def test_rejects_empty_score_description() -> None:
    payload = valid_payload()
    payload["dimensions"][0]["scoring_guide"]["2"] = " "

    with pytest.raises(ValidationError, match="must not be empty"):
        Rubric.model_validate(payload)


def test_rejects_out_of_range_score_key() -> None:
    payload = valid_payload()
    payload["dimensions"][0]["scoring_guide"]["6"] = "超过最高分"

    with pytest.raises(ValidationError, match="outside 1-5"):
        Rubric.model_validate(payload)


def test_rejects_duplicate_dimension_ids() -> None:
    payload = valid_payload()
    second = dict(payload["dimensions"][0])
    second["name"] = "完整性"
    payload["dimensions"].append(second)

    with pytest.raises(ValidationError, match="dimension id must be unique"):
        Rubric.model_validate(payload)


def test_normalizes_score_keys_with_chinese_suffix() -> None:
    payload = valid_payload()
    payload["dimensions"][0]["scoring_guide"] = {
        "1分": "核心结论错误",
        "2分": "多处关键信息错误",
        "3分": "主要结论正确但遗漏明显",
        "4分": "结论正确且只有轻微遗漏",
        "5分": "完全准确且可执行",
    }

    rubric = Rubric.model_validate(payload)

    assert list(rubric.dimensions[0].scoring_guide) == ["1", "2", "3", "4", "5"]
