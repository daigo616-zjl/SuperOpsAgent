"""多 Agent 诊断编排的核心数据模型。

星型拓扑下的消息契约：
- Supervisor 给 Hypothesizer 观测、给取证 Agent 发 Directive、
  给 Adjudicator 筛选后的证据子集、给 Reporter claim 白名单；
- 各角色产出（EvidenceCard / AdjudicationDecision）只回到 Supervisor。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

HypothesisStatus = Literal["active", "confirmed", "ruled_out"]
EvidenceDomain = Literal["metrics", "logs", "knowledge"]
EvidencePolarity = Literal["supports", "refutes", "neutral"]


class DiagnosisContext(BaseModel):
    """贯穿诊断流程且由服务层统一解析的上下文。"""

    service_name: str = Field(min_length=1, description="本次诊断的目标服务")


class Hypothesis(BaseModel):
    """候选根因假设，带鉴别性预期证据。"""

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    prior: float = Field(ge=0.0, le=1.0, default=0.5)
    expected_support: list[str] = Field(default_factory=list)
    expected_refuting: list[str] = Field(default_factory=list)
    status: HypothesisStatus = "active"
    posterior: float | None = Field(default=None, ge=0.0, le=1.0)
    ruled_out_by: list[str] = Field(
        default_factory=list,
        description="淘汰该假设的 claim_id 列表，必须指向 Evidence Store 中已存在的 claim",
    )


class ClaimProvenance(BaseModel):
    """证据出处的机械可校验记录。"""

    tool_name: str = Field(min_length=1)
    args_digest: str = Field(min_length=1, description="工具调用参数摘要（哈希或压缩文本）")
    output_path: str | None = None
    excerpt: str = Field(min_length=1, max_length=2000)


class EvidenceClaim(BaseModel):
    """单条可核查的证据判断。"""

    claim_id: str = Field(min_length=1, pattern=r"^ev-[a-zA-Z0-9_-]+$")
    statement: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    polarity: EvidencePolarity = "neutral"
    hypothesis_ids: list[str] = Field(default_factory=list)
    provenance: ClaimProvenance


class EvidenceCard(BaseModel):
    """取证 Agent 的唯一产出物。"""

    card_id: str = Field(min_length=1)
    domain: EvidenceDomain
    directive_id: str = Field(min_length=1)
    round: int = Field(ge=0)
    claims: list[EvidenceClaim] = Field(min_length=1)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> "EvidenceCard":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"card 内 claim_id 重复: {self.card_id}")
        return self


class Directive(BaseModel):
    """Supervisor 派发给单个取证 Agent 的任务。"""

    id: str = Field(min_length=1)
    target_domain: EvidenceDomain
    objective: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=4, ge=1, le=10)


class Elimination(BaseModel):
    """Adjudicator 对单条假设的淘汰裁决。"""

    hypothesis_id: str = Field(min_length=1)
    ruled_out_by: list[str] = Field(min_length=1, description="支撑淘汰的 claim_id，禁止为空")
    reason: str = Field(min_length=1)


class AdjudicationDecision(BaseModel):
    """评审结果，只回给 Supervisor。"""

    eliminations: list[Elimination] = Field(default_factory=list)
    new_directives: list[Directive] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    converged: bool = False
    converged_hypothesis_id: str | None = None

    @model_validator(mode="after")
    def validate_convergence(self) -> "AdjudicationDecision":
        if self.converged and not self.converged_hypothesis_id:
            raise ValueError("converged 为 true 时必须给出 converged_hypothesis_id")
        return self


class SupervisorDecision(BaseModel):
    """确定性路由的中间产物（不调 LLM，仅用于事件与测试断言）。"""

    action: Literal["hypothesize", "dispatch", "adjudicate", "converge"]
    directives: list[Directive] = Field(default_factory=list)
    converged_hypothesis_id: str | None = None
    reason: str = ""


class BudgetLedger(BaseModel):
    """三重预算：轮数 / LLM 迭代 / 墙钟。"""

    round: int = 0
    max_rounds: int = Field(default=6, ge=1)
    invocations: int = 0
    max_invocations: int = Field(default=60, ge=1)
    started_at: datetime = Field(default_factory=datetime.now)
    max_wall_seconds: float = Field(default=300.0, gt=0)
    min_dispatch_wall_seconds: float = Field(default=90.0, gt=0)
    investigation_wall_seconds: float = Field(default=150.0, gt=0)

    def remaining_rounds(self) -> int:
        return self.max_rounds - self.round

    def remaining_invocations(self) -> int:
        return self.max_invocations - self.invocations

    def remaining_wall_seconds(self, now: datetime | None = None) -> float:
        return max(0.0, self.max_wall_seconds - self.elapsed_seconds(now))

    def elapsed_seconds(self, now: datetime | None = None) -> float:
        current = now or datetime.now()
        return max(0.0, (current - self.started_at).total_seconds())

    def wall_exhausted(self, now: datetime | None = None) -> bool:
        return self.elapsed_seconds(now) >= self.max_wall_seconds

    def exhausted(self, now: datetime | None = None) -> bool:
        return self.remaining_rounds() <= 0 or self.wall_exhausted(now)
