"""AIOps Evidence Store 仓储。

证据卡与证据 claim 严格 append-only（只 INSERT，冲突时静默跳过以支持
幂等重放），无 UPDATE/DELETE；诊断会话行仅在收尾时允许一次状态落库。
"""

import json
from typing import Any

from sqlalchemy import text

from app.agent.aiops.diagnosis_models import EvidenceCard
from app.core.postgres import postgres_manager


class EvidenceRepository:
    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        return dict(row._mapping)

    def start_session(
        self, session_id: str, service_name: str,
        scenario_id: str | None = None, budget_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text("""
                insert into aiops_diagnosis_sessions
                    (session_id, service_name, scenario_id, budget_snapshot)
                values (:session_id, :service_name, :scenario_id,
                        cast(:budget_snapshot as jsonb))
                on conflict (session_id) do update set scenario_id = excluded.scenario_id
                returning session_id, service_name, scenario_id, status,
                          budget_snapshot, started_at, finished_at
            """), {
                "session_id": session_id,
                "service_name": service_name,
                "scenario_id": scenario_id,
                "budget_snapshot": json.dumps(budget_snapshot or {}, ensure_ascii=False),
            }).one()
            return self._row(row)

    def finish_session(
        self, session_id: str, status: str,
        final_hypothesis_id: str | None = None,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> None:
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                update aiops_diagnosis_sessions
                set status = :status,
                    final_hypothesis_id = :final_hypothesis_id,
                    budget_snapshot = cast(:budget_snapshot as jsonb),
                    finished_at = now()
                where session_id = :session_id and finished_at is null
            """), {
                "session_id": session_id,
                "status": status,
                "final_hypothesis_id": final_hypothesis_id,
                "budget_snapshot": json.dumps(budget_snapshot or {}, ensure_ascii=False),
            })

    def append_evidence_card(
        self, session_id: str, card: EvidenceCard, directive: dict[str, Any] | None = None,
    ) -> None:
        """卡片与 claims 在同一事务内追加；重复 card_id/claim_id 幂等跳过。"""
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                insert into aiops_evidence_cards
                    (card_id, session_id, round, domain, directive, summary)
                values (:card_id, :session_id, :round, :domain,
                        cast(:directive as jsonb), :summary)
                on conflict (card_id) do nothing
            """), {
                "card_id": card.card_id,
                "session_id": session_id,
                "round": card.round,
                "domain": card.domain,
                "directive": json.dumps(directive or {}, ensure_ascii=False),
                "summary": card.summary,
            })
            for claim in card.claims:
                connection.execute(text("""
                    insert into aiops_evidence_claims
                        (claim_id, card_id, statement, confidence, polarity,
                         hypothesis_ids, provenance)
                    values (:claim_id, :card_id, :statement, :confidence, :polarity,
                            :hypothesis_ids, cast(:provenance as jsonb))
                    on conflict (claim_id) do nothing
                """), {
                    "claim_id": claim.claim_id,
                    "card_id": card.card_id,
                    "statement": claim.statement,
                    "confidence": claim.confidence,
                    "polarity": claim.polarity,
                    "hypothesis_ids": list(claim.hypothesis_ids),
                    "provenance": claim.provenance.model_dump_json(),
                })

    def list_cards(self, session_id: str) -> list[dict[str, Any]]:
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select card_id, session_id, round, domain, directive, summary, created_at
                from aiops_evidence_cards
                where session_id = :session_id
                order by round, id
            """), {"session_id": session_id}).all()
            return [self._row(row) for row in rows]

    def list_claims(self, session_id: str) -> list[dict[str, Any]]:
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select c.claim_id, c.card_id, c.statement, c.confidence, c.polarity,
                       c.hypothesis_ids, c.provenance, c.created_at
                from aiops_evidence_claims c
                join aiops_evidence_cards k on k.card_id = c.card_id
                where k.session_id = :session_id
                order by k.round, c.id
            """), {"session_id": session_id}).all()
            return [self._row(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with postgres_manager.engine.connect() as connection:
            row = connection.execute(text("""
                select session_id, service_name, scenario_id, status,
                       final_hypothesis_id, budget_snapshot, started_at, finished_at
                from aiops_diagnosis_sessions
                where session_id = :session_id
            """), {"session_id": session_id}).one_or_none()
            return self._row(row) if row else None


evidence_repository = EvidenceRepository()
