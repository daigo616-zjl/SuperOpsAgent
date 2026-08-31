-- AIOps 多 Agent 诊断 Evidence Store（append-only：证据表只增不改不删）

create table if not exists aiops_diagnosis_sessions (
    id bigint generated always as identity primary key,
    session_id uuid not null unique,
    service_name text not null,
    scenario_id text,
    status text not null default 'running'
        check (status in ('running', 'completed', 'failed', 'budget_exhausted')),
    final_hypothesis_id text,
    budget_snapshot jsonb not null default '{}'::jsonb,
    started_at timestamptz not null default now(),
    finished_at timestamptz
);

create index if not exists aiops_diagnosis_sessions_service_started_idx
    on aiops_diagnosis_sessions (service_name, started_at desc);

create table if not exists aiops_evidence_cards (
    id bigint generated always as identity primary key,
    card_id uuid not null unique,
    session_id uuid not null references aiops_diagnosis_sessions(session_id) on delete cascade,
    round integer not null check (round >= 0),
    domain text not null check (domain in ('metrics', 'logs', 'knowledge')),
    directive jsonb not null default '{}'::jsonb,
    summary text not null default '',
    created_at timestamptz not null default now()
);

create index if not exists aiops_evidence_cards_session_idx
    on aiops_evidence_cards (session_id, round, id);

create table if not exists aiops_evidence_claims (
    id bigint generated always as identity primary key,
    claim_id text not null unique,
    card_id uuid not null references aiops_evidence_cards(card_id) on delete cascade,
    statement text not null,
    confidence numeric(4, 3) not null check (confidence >= 0 and confidence <= 1),
    polarity text not null check (polarity in ('supports', 'refutes', 'neutral')),
    hypothesis_ids text[] not null default '{}',
    provenance jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists aiops_evidence_claims_card_idx
    on aiops_evidence_claims (card_id, id);
create index if not exists aiops_evidence_claims_hypothesis_idx
    on aiops_evidence_claims using gin (hypothesis_ids);
