import math
import re
from collections import Counter
from decimal import Decimal
from typing import Dict, Iterable, List, Sequence, Set

from pydantic import BaseModel, Field

from .schemas import Rubric, RubricDimension


class ValidationIssue(BaseModel):
    severity: str
    code: str
    message: str
    dimensions: List[str] = Field(default_factory=list)
    evidence: str = ""
    action: str = ""
    dimension_name: str = ""
    example_type: str = ""
    example_index: int = 0
    task_keyword_overlap: float = 0.0
    domain_term_hits: List[str] = Field(default_factory=list)
    dimension_term_hits: List[str] = Field(default_factory=list)
    suggestion: str = ""


class ValidationReport(BaseModel):
    weight_sum: float
    error_count: int
    warning_count: int
    info_count: int
    issues: List[ValidationIssue]
    checked_example_count: int = 0
    low_literal_count: int = 0
    relevance_keep_count: int = 0
    relevance_rewrite_count: int = 0
    relevance_delete_count: int = 0


SYNONYMS = {
    "准确性": "正确性",
    "准确度": "正确性",
    "完整程度": "完整性",
    "清晰度": "清晰性",
    "关联性": "相关性",
    "相关程度": "相关性",
    "流畅度": "流畅性",
    "一致程度": "一致性",
    "七天无理由": "7天无理由",
    "退回原支付方式": "原路退回",
}

GENERIC_TERMS = {
    "评测",
    "评估",
    "模型",
    "任务",
    "目标",
    "结果",
    "输出",
    "内容",
    "样例",
    "要求",
    "进行",
    "是否",
    "一个",
}

DOMAIN_PROFILES = {
    "refund_customer_service": {
        "label": "退款或售后",
        "suggestion_terms": "订单、审核、凭证、到账时效或处理步骤",
        "triggers": {
            "退款",
            "退货",
            "售后",
            "refund",
        },
        "core_terms": {
            "退款",
            "退货",
            "售后",
            "订单",
            "申请",
            "审核",
            "到账",
            "原路退回",
            "凭证",
            "商品",
            "签收",
            "拒绝",
            "申诉",
            "客服",
            "平台介入",
        },
        "policy_terms": {
            "7天无理由",
            "无理由退款",
            "定制商品",
            "虚拟商品",
            "商品完好",
            "质量问题",
            "运费",
            "时效",
            "工作日",
            "自然日",
        },
        "action_terms": {
            "点击",
            "上传",
            "提交",
            "联系",
            "等待审核",
            "补充材料",
            "申请售后",
            "申请退款",
            "操作步骤",
            "下一步",
        },
        "adjacent_terms": {
            "优惠券",
            "积分",
            "促销活动",
            "价格保护",
        },
        "unrelated_terms": {
            "修改头像",
            "个人资料页",
            "修改昵称",
            "重置密码",
            "登录密码",
        },
    }
}

DIMENSION_PROFILES = {
    "policy_accuracy": {
        "triggers": {"政策", "准确", "正确", "规则"},
        "terms": {
            "政策",
            "规则",
            "条件",
            "时效",
            "7天",
            "定制商品",
            "无理由退款",
            "到账时间",
            "适用",
            "不支持",
            "错误",
            "准确",
        },
        "generic": False,
    },
    "information_completeness": {
        "triggers": {"完整", "信息"},
        "terms": {
            "流程",
            "材料",
            "入口",
            "凭证",
            "审核",
            "到账",
            "注意事项",
            "遗漏",
            "必要",
            "完整",
        },
        "generic": False,
    },
    "resolution": {
        "triggers": {"解决", "推动", "行动", "可操作"},
        "terms": {
            "下一步",
            "点击",
            "上传",
            "提交",
            "申诉",
            "人工客服",
            "替代方案",
            "平台介入",
            "操作",
            "推进",
        },
        "generic": False,
    },
    "courtesy": {
        "triggers": {"礼貌", "专业", "语气", "态度"},
        "terms": {
            "您好",
            "抱歉",
            "理解",
            "感谢",
            "请",
            "随时联系",
            "生硬",
            "指责",
            "不耐烦",
            "亲切",
            "专业",
        },
        "generic": True,
    },
    "clarity": {
        "triggers": {"清晰", "表达", "条理", "可读"},
        "terms": {
            "分点",
            "步骤",
            "说明",
            "清晰",
            "歧义",
            "工作日",
            "自然日",
            "条理",
            "混乱",
            "明确",
        },
        "generic": True,
    },
}


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    for source, target in SYNONYMS.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def character_ngrams(text: str, sizes: Sequence[int] = (2, 3)) -> Counter:
    compact = normalize_text(text)
    grams: Counter = Counter()
    for size in sizes:
        if len(compact) < size:
            continue
        grams.update(compact[index : index + size] for index in range(len(compact) - size + 1))
    if not grams and compact:
        grams[compact] = 1
    return grams


def cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[item] * right[item] for item in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def text_similarity(left: str, right: str) -> float:
    if normalize_text(left) == normalize_text(right):
        return 1.0
    return cosine_similarity(character_ngrams(left), character_ngrams(right))


def extract_terms(text: str) -> Set[str]:
    lowered = text.lower()
    latin_terms = set(re.findall(r"[a-z0-9][a-z0-9.+_-]{1,}", lowered))
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese_terms: Set[str] = set()
    for chunk in chinese_chunks:
        for size in (2, 3, 4):
            chinese_terms.update(
                chunk[index : index + size] for index in range(max(0, len(chunk) - size + 1))
            )
    return {term for term in latin_terms | chinese_terms if term not in GENERIC_TERMS}


def relevance_score(sample: str, context_terms: Set[str]) -> float:
    sample_terms = extract_terms(sample)
    if not sample_terms or not context_terms:
        return 0.0
    overlap = sample_terms & context_terms
    return len(overlap) / min(len(sample_terms), len(context_terms))


def find_phrase_hits(text: str, terms: Iterable[str]) -> List[str]:
    normalized = normalize_text(text)
    return sorted(
        term
        for term in terms
        if normalize_text(term) and normalize_text(term) in normalized
    )


def infer_domain_profile(context: str) -> Dict:
    for profile in DOMAIN_PROFILES.values():
        if find_phrase_hits(context, profile["triggers"]):
            return profile
    return {}


def domain_label(domain_profile: Dict) -> str:
    return str(domain_profile.get("label") or "当前任务")


def domain_suggestion_terms(domain_profile: Dict) -> str:
    return str(domain_profile.get("suggestion_terms") or "任务中的关键对象、约束或处理步骤")


def infer_dimension_profile(dimension: RubricDimension) -> Dict:
    context = f"{dimension.name} {dimension.description}"
    matched_profiles = [
        profile
        for profile in DIMENSION_PROFILES.values()
        if find_phrase_hits(context, profile["triggers"])
    ]
    if not matched_profiles:
        return {}
    return {
        "terms": set().union(*(profile["terms"] for profile in matched_profiles)),
        "generic": all(profile["generic"] for profile in matched_profiles),
    }


def make_relevance_issue(
    dimension: RubricDimension,
    label: str,
    example_index: int,
    score: float,
    domain_hits: List[str],
    dimension_hits: List[str],
    action: str,
    suggestion: str,
) -> ValidationIssue:
    labels = {
        "keep": ("pass", "建议保留"),
        "rewrite": ("info", "建议场景化改写"),
        "delete": ("error", "建议删除"),
    }
    severity, action_label = labels[action]
    hit_parts = []
    if domain_hits:
        hit_parts.append(f"领域词：{'、'.join(domain_hits)}")
    if dimension_hits:
        hit_parts.append(f"维度词：{'、'.join(dimension_hits)}")
    evidence = f"字面重合 {score:.0%}"
    if hit_parts:
        evidence += "；" + "；".join(hit_parts)

    return ValidationIssue(
        severity=severity,
        code=f"example_relevance_{action}",
        message=f"“{dimension.name}”的{label}样例 {example_index}：{action_label}。",
        dimensions=[dimension.id],
        evidence=evidence,
        action=action,
        dimension_name=dimension.name,
        example_type=label,
        example_index=example_index,
        task_keyword_overlap=score,
        domain_term_hits=domain_hits,
        dimension_term_hits=dimension_hits,
        suggestion=suggestion,
    )


def validate_weights(rubric: Rubric) -> List[ValidationIssue]:
    total = sum(Decimal(str(item.weight)) for item in rubric.dimensions)
    issues: List[ValidationIssue] = []
    tolerance = Decimal("0.0001")

    if total > Decimal("1.0") + tolerance:
        issues.append(
            ValidationIssue(
                severity="error",
                code="weight_overflow",
                message=f"权重总和为 {total}，超过上限 1.0。",
                dimensions=[item.id for item in rubric.dimensions],
                evidence=f"超出 {total - Decimal('1.0')}",
            )
        )
    elif total < Decimal("1.0") - tolerance:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="weight_underflow",
                message=f"权重总和为 {total}，尚未分配到 1.0。",
                dimensions=[item.id for item in rubric.dimensions],
                evidence=f"剩余 {Decimal('1.0') - total}",
            )
        )
    return issues


def validate_duplicate_dimensions(
    dimensions: Sequence[RubricDimension],
    threshold: float = 0.72,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for index, left in enumerate(dimensions):
        for right in dimensions[index + 1 :]:
            name_score = text_similarity(left.name, right.name)
            description_score = text_similarity(left.description, right.description)
            combined = name_score * 0.65 + description_score * 0.35
            if name_score == 1.0 or combined >= threshold:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="duplicate_dimension",
                        message=f"“{left.name}”与“{right.name}”可能高度重复。",
                        dimensions=[left.id, right.id],
                        evidence=f"本地文本相似度 {combined:.0%}",
                    )
                )
    return issues


def validate_examples(
    rubric: Rubric,
    task_description: str,
    model_type: str,
    evaluation_goal: str,
) -> List[ValidationIssue]:
    context = " ".join([task_description, model_type, evaluation_goal, rubric.task_summary])
    context_terms = extract_terms(context)
    domain_profile = infer_domain_profile(context)
    domain_terms = set().union(
        domain_profile.get("core_terms", set()),
        domain_profile.get("policy_terms", set()),
        domain_profile.get("action_terms", set()),
    )
    issues: List[ValidationIssue] = []

    for dimension in rubric.dimensions:
        dimension_profile = infer_dimension_profile(dimension)
        dimension_terms = dimension_profile.get("terms", set())
        if not dimension.positive_examples:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_positive_example",
                    message=f"“{dimension.name}”缺少正向样例。",
                    dimensions=[dimension.id],
                )
            )
        if not dimension.negative_examples:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_negative_example",
                    message=f"“{dimension.name}”缺少负向样例。",
                    dimensions=[dimension.id],
                )
            )

        for label, examples in (
            ("正向", dimension.positive_examples),
            ("负向", dimension.negative_examples),
        ):
            for example_index, example in enumerate(examples, start=1):
                score = relevance_score(example, context_terms)
                if len(normalize_text(example)) < 8 or score >= 0.06:
                    continue

                domain_hits = find_phrase_hits(example, domain_terms)
                dimension_hits = find_phrase_hits(example, dimension_terms)
                action_hits = find_phrase_hits(
                    example,
                    domain_profile.get("action_terms", set()),
                )
                adjacent_hits = find_phrase_hits(
                    example,
                    domain_profile.get("adjacent_terms", set()),
                )
                unrelated_hits = find_phrase_hits(
                    example,
                    domain_profile.get("unrelated_terms", set()),
                )

                if unrelated_hits:
                    action = "delete"
                    suggestion = (
                        "该样例命中了明显无关场景词："
                        f"{'、'.join(unrelated_hits)}。建议替换为当前评测任务的样例。"
                    )
                elif adjacent_hits and not domain_hits:
                    action = "rewrite"
                    suggestion = (
                        f"样例涉及相邻业务：{'、'.join(adjacent_hits)}，"
                        f"建议补充与{domain_label(domain_profile)}流程的直接关系。"
                    )
                elif dimension_profile.get("generic") and len(domain_hits) < 2:
                    action = "rewrite"
                    suggestion = (
                        f"该样例能体现当前维度，但{domain_label(domain_profile)}场景较弱。"
                        f"建议补充{domain_suggestion_terms(domain_profile)}。"
                    )
                elif domain_hits and (dimension_hits or not dimension_profile):
                    action = "keep"
                    suggestion = "命中了领域词或维度词，属于合理子场景，无需修改。"
                elif action_hits:
                    action = "keep"
                    suggestion = "命中了明确操作词，能够体现问题推进，建议保留。"
                elif dimension_hits:
                    action = "rewrite"
                    suggestion = (
                        "方向符合当前维度，但任务场景不够明确。"
                        f"建议加入{domain_suggestion_terms(domain_profile)}等上下文。"
                    )
                else:
                    action = "rewrite"
                    suggestion = (
                        "该样例与原始任务、领域词和维度词均缺少明显联系，"
                        "建议改写得更贴近当前任务。"
                    )

                issues.append(
                    make_relevance_issue(
                        dimension=dimension,
                        label=label,
                        example_index=example_index,
                        score=score,
                        domain_hits=domain_hits,
                        dimension_hits=dimension_hits,
                        action=action,
                        suggestion=suggestion,
                    )
                )

        for positive in dimension.positive_examples:
            for negative in dimension.negative_examples:
                similarity = text_similarity(positive, negative)
                if similarity >= 0.88:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="example_conflict",
                            message=f"“{dimension.name}”存在高度相似的正向和负向样例。",
                            dimensions=[dimension.id],
                            evidence=f"本地文本相似度 {similarity:.0%}",
                        )
                    )
                    break

        score_keys = {normalize_text(key) for key in dimension.scoring_guide}
        minimum = str(rubric.score_scale.min)
        maximum = str(rubric.score_scale.max)
        if not any(minimum in key for key in score_keys) or not any(
            maximum in key for key in score_keys
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="score_range_incomplete",
                    message=f"“{dimension.name}”的评分说明未明确覆盖最低分和最高分。",
                    dimensions=[dimension.id],
                    evidence=f"期望覆盖 {minimum} 分与 {maximum} 分",
                )
            )
    return issues


def validate_rubric(
    rubric: Rubric,
    task_description: str,
    model_type: str,
    evaluation_goal: str,
) -> ValidationReport:
    issues = [
        *validate_weights(rubric),
        *validate_duplicate_dimensions(rubric.dimensions),
        *validate_examples(rubric, task_description, model_type, evaluation_goal),
    ]
    total = sum(Decimal(str(item.weight)) for item in rubric.dimensions)
    counts: Dict[str, int] = Counter(issue.severity for issue in issues)
    relevance_issues = [
        issue for issue in issues if issue.code.startswith("example_relevance_")
    ]
    action_counts: Dict[str, int] = Counter(issue.action for issue in relevance_issues)
    return ValidationReport(
        weight_sum=float(total),
        error_count=counts.get("error", 0),
        warning_count=counts.get("warning", 0),
        info_count=counts.get("info", 0),
        issues=issues,
        checked_example_count=sum(
            len(item.positive_examples) + len(item.negative_examples)
            for item in rubric.dimensions
        ),
        low_literal_count=len(relevance_issues),
        relevance_keep_count=action_counts.get("keep", 0),
        relevance_rewrite_count=action_counts.get("rewrite", 0),
        relevance_delete_count=action_counts.get("delete", 0),
    )
