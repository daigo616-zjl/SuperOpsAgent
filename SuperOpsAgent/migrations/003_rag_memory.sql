-- RAG 长期记忆：强事实结构化存储 + 抽取任务 Outbox

-- 强事实：零容错结构化记忆，召回优先级最高
create table if not exists rag_memory_facts (
    id bigint generated always as identity primary key,
    memory_id uuid not null unique,
    user_id text not null,
    content text not null,
    subject text not null default '',
    keywords text[] not null default '{}',
    content_hash char(64) not null,
    status text not null default 'active'
        check (status in ('active', 'superseded')),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- 同一用户同一内容只保留一条 active 记录（幂等锚点）
create unique index if not exists rag_memory_facts_user_hash_active_idx
    on rag_memory_facts (user_id, content_hash) where status = 'active';

create index if not exists rag_memory_facts_user_status_idx
    on rag_memory_facts (user_id, status, updated_at desc);
create index if not exists rag_memory_facts_keywords_idx
    on rag_memory_facts using gin (keywords);

-- 长期记忆抽取任务 Outbox：请求内同步入队，worker 异步消费
create table if not exists rag_memory_jobs (
    id bigint generated always as identity primary key,
    job_id uuid not null unique,
    session_id text not null,
    user_id text not null,
    user_message text not null,
    assistant_message text not null default '',
    summary text not null default '',
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'done', 'dead')),
    attempts integer not null default 0,
    lease_until timestamptz,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists rag_memory_jobs_claim_idx
    on rag_memory_jobs (status, id) where status in ('pending', 'processing');
