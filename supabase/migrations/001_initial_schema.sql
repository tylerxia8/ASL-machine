-- ASL Pilot schema (Supabase / Postgres)
-- Link auth.users.id to profiles.id

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  created_at timestamptz default now()
);

create table if not exists public.practice_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  started_at timestamptz default now(),
  ended_at timestamptz,
  signs_practiced int default 0,
  signs_passed int default 0
);

create table if not exists public.attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  session_id uuid references public.practice_sessions (id) on delete set null,
  sign_id text not null,
  outcome text not null check (outcome in ('pass', 'fail', 'retry')),
  confidence real,
  predicted_label text,
  attempt_number int default 1,
  created_at timestamptz default now()
);

create table if not exists public.sign_mastery (
  user_id uuid not null references public.profiles (id) on delete cascade,
  sign_id text not null,
  mastered boolean default false,
  consecutive_passes int default 0,
  total_attempts int default 0,
  total_passes int default 0,
  last_practiced_at timestamptz,
  primary key (user_id, sign_id)
);

alter table public.profiles enable row level security;
alter table public.practice_sessions enable row level security;
alter table public.attempts enable row level security;
alter table public.sign_mastery enable row level security;

create policy "Users read own profile" on public.profiles
  for select using (auth.uid() = id);

create policy "Users update own profile" on public.profiles
  for update using (auth.uid() = id);

create policy "Users manage own sessions" on public.practice_sessions
  for all using (auth.uid() = user_id);

create policy "Users manage own attempts" on public.attempts
  for all using (auth.uid() = user_id);

create policy "Users manage own mastery" on public.sign_mastery
  for all using (auth.uid() = user_id);
