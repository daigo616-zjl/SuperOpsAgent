-- 最小用户体系：长期记忆主体（user_id）的注册与校验来源

create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    display_name text not null default '',
    created_at timestamptz not null default now()
);
