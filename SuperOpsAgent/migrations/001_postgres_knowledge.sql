create table if not exists knowledge_documents (
    id bigint generated always as identity primary key,
    public_id uuid not null unique,
    title text not null check (char_length(title) between 1 and 500),
    source_path text not null,
    content text not null,
    content_hash char(64) not null,
    version bigint not null default 1 check (version > 0),
    deleted_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists knowledge_documents_active_source_uidx
    on knowledge_documents (lower(source_path)) where deleted_at is null;
create index if not exists knowledge_documents_active_updated_idx
    on knowledge_documents (updated_at desc, id desc) where deleted_at is null;

create table if not exists knowledge_index_registry (
    document_id bigint primary key references knowledge_documents(id) on delete cascade,
    content_hash char(64) not null,
    document_version bigint not null,
    index_version uuid not null,
    chunk_ids jsonb not null default '[]'::jsonb,
    chunk_count integer not null check (chunk_count >= 0),
    status text not null default 'ready' check (status in ('ready', 'repairing', 'failed')),
    last_error text,
    indexed_at timestamptz not null default now(),
    checked_at timestamptz,
    updated_at timestamptz not null default now()
);

alter table knowledge_index_registry
    add column if not exists checked_at timestamptz;

create table if not exists knowledge_index_jobs (
    id bigint generated always as identity primary key,
    public_id uuid not null unique,
    document_id bigint not null references knowledge_documents(id) on delete cascade,
    event_type text not null check (event_type in ('upsert', 'delete', 'repair')),
    document_version bigint not null,
    content_hash char(64) not null,
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'retry', 'succeeded', 'superseded', 'dead')),
    attempts integer not null default 0 check (attempts >= 0),
    available_at timestamptz not null default now(),
    locked_at timestamptz,
    worker_id text,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz,
    unique (document_id, document_version, event_type)
);

create index if not exists knowledge_index_jobs_claim_idx
    on knowledge_index_jobs (available_at, id)
    where status in ('pending', 'retry');
create index if not exists knowledge_index_jobs_document_status_idx
    on knowledge_index_jobs (document_id, status, id);
