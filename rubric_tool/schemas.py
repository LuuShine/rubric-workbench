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
