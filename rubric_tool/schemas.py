import re
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScoreScale(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min: int = Field(default=1)
    max: int = Field(default=5)

    @model_validator(mode="after")
    def validate_range(self) -> "ScoreScale":
        if self.max <= self.min:
            raise ValueError("score_scale.max must be greater than score_scale.min")
        return self


class RubricDimension(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str
    weight: float = Field(ge=0)
    positive_examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    scoring_guide: Dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "name", "description")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("positive_examples", "negative_examples")
    @classmethod
    def clean_examples(cls, values: List[str]) -> List[str]:
        return [value.strip() for value in values if value and value.strip()]

    @field_validator("scoring_guide")
    @classmethod
    def require_non_empty_scoring_text(cls, values: Dict[str, str]) -> Dict[str, str]:
        cleaned = {str(key).strip(): str(value).strip() for key, value in values.items()}
        if not cleaned:
            raise ValueError("scoring_guide must not be empty")
        empty_keys = [key for key, value in cleaned.items() if not key or not value]
        if empty_keys:
            raise ValueError("scoring_guide keys and descriptions must not be empty")
        return cleaned


class Rubric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_summary: str
    score_scale: ScoreScale = Field(default_factory=ScoreScale)
    dimensions: List[RubricDimension] = Field(min_length=1)

    @field_validator("task_summary")
    @classmethod
    def require_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task_summary must not be empty")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> "Rubric":
        ids = [dimension.id for dimension in self.dimensions]
        duplicated_ids = sorted({item for item in ids if ids.count(item) > 1})
        if duplicated_ids:
            raise ValueError(f"dimension id must be unique: {', '.join(duplicated_ids)}")

        expected_scores = set(range(self.score_scale.min, self.score_scale.max + 1))
        for dimension in self.dimensions:
            normalized_guide: Dict[str, str] = {}
            for raw_key, description in dimension.scoring_guide.items():
                score = parse_score_key(raw_key)
                if score is None:
                    raise ValueError(
                        f"{dimension.id}.scoring_guide has invalid score key: {raw_key}"
                    )
                if score < self.score_scale.min or score > self.score_scale.max:
                    raise ValueError(
                        f"{dimension.id}.scoring_guide score {score} is outside "
                        f"{self.score_scale.min}-{self.score_scale.max}"
                    )
                normalized_key = str(score)
                if normalized_key in normalized_guide:
                    raise ValueError(
                        f"{dimension.id}.scoring_guide has duplicate score key: {score}"
                    )
                normalized_guide[normalized_key] = description

            present_scores = {int(score) for score in normalized_guide}
            missing_scores = sorted(expected_scores - present_scores)
            if missing_scores:
                missing = ", ".join(str(score) for score in missing_scores)
                raise ValueError(
                    f"{dimension.id}.scoring_guide must cover every score in "
                    f"{self.score_scale.min}-{self.score_scale.max}; missing {missing}"
                )
            dimension.scoring_guide = {
                str(score): normalized_guide[str(score)] for score in sorted(expected_scores)
            }
        return self


def parse_score_key(value: str) -> int | None:
    match = re.fullmatch(r"\s*(-?\d+)\s*分?\s*", str(value))
    if not match:
        return None
    return int(match.group(1))
