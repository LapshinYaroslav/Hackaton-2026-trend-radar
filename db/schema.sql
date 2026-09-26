-- Collector cache. Dates are DATE (UTC calendar days, YYYY-MM-DD).
-- features is filled by compute_features (Yaroslav); the collector only reads it to skip Search #2.

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    tech_key        TEXT        NOT NULL,
    candidate_id    TEXT,
    query_id        TEXT,
    published_at    DATE        NOT NULL,
    source          TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    url             TEXT        NOT NULL,
    language        TEXT        NOT NULL DEFAULT 'en',
    trust_level     TEXT        NOT NULL,
    organizations   TEXT[]      NOT NULL DEFAULT '{}',
    body            TEXT        NOT NULL DEFAULT '',
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_source_type_chk CHECK (source_type IN (
        'paper', 'preprint', 'patent', 'news', 'press_release',
        'product', 'report', 'standard', 'blog'
    )),
    CONSTRAINT documents_trust_level_chk CHECK (trust_level IN ('high', 'medium', 'low')),
    CONSTRAINT documents_tech_url_uq UNIQUE (tech_key, url)
);

CREATE INDEX IF NOT EXISTS documents_tech_key_idx ON documents (tech_key);
CREATE INDEX IF NOT EXISTS documents_published_at_idx ON documents (published_at);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
CREATE INDEX IF NOT EXISTS documents_candidate_id_idx ON documents (candidate_id);

-- period, а не window: WINDOW — зарезервированное слово PostgreSQL.
-- В контракте (SourceTotal, compute_features) поле по-прежнему называется window.
CREATE TABLE IF NOT EXISTS source_totals (
    source          TEXT        NOT NULL,
    period          TEXT        NOT NULL,
    n_total         INTEGER     NOT NULL,
    available       BOOLEAN     NOT NULL DEFAULT TRUE,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Чем посчитан корпус. Совпадения этих двух полей достаточно, чтобы переиспользовать
    -- строку: сменился фильтр типа или сам запрос за итогом — числа несравнимы.
    type_filter     TEXT,
    totals_signature TEXT,
    PRIMARY KEY (source, period),
    -- before и now нужны поиску №1, годовые — поиску №2, all — имя сложенного итога.
    CONSTRAINT source_totals_period_chk CHECK (
        period IN ('before', 'now', 'all', '2020', '2021', '2022', '2023', '2024', '2025')
    )
);

-- Счётчики поиска №2 (pipeline.md 0.2). Одна строка на
-- (технология, термины, источник, окно, семантика запроса).
-- terms_hash в ключе: уточнили термины — старые числа вернуться не должны. Фильтра
-- типа в нём нет, это константа OpenAlex; семантика запроса живёт в query_variant.
-- query_variant в ключе: один источник опрашивается несколькими способами —
-- фразой с type:article, той же строкой без кавычек, без фильтра типа. Числа разных
-- способов несравнимы, и без этого поля вторые затирали бы первые.
CREATE TABLE IF NOT EXISTS counters (
    tech_key        TEXT        NOT NULL,
    terms_hash      TEXT        NOT NULL,
    source          TEXT        NOT NULL,
    period          TEXT        NOT NULL,
    query_variant   TEXT        NOT NULL DEFAULT '',
    date_from       DATE        NOT NULL,
    date_to         DATE        NOT NULL,
    n               INTEGER     NOT NULL,
    type_filter     TEXT,
    candidate_id    TEXT,
    query_id        TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tech_key, terms_hash, source, period, query_variant),
    -- Хранятся годовые счётчики и окно prev6, все непересекающиеся. Рабочие окна
    -- all, now, before и recent24 складываются из них в model/counters.py и в базу
    -- не пишутся. prev6 — предыдущая шестилетка, 2014-09-01…2020-09-01; корпусного
    -- итога за неё нет, признак share_prev6 нормировки не требует.
    CONSTRAINT counters_period_chk CHECK (
        period IN ('prev6', '2020', '2021', '2022', '2023', '2024', '2025')
    ),
    CONSTRAINT counters_n_chk CHECK (n >= 0)
);

CREATE INDEX IF NOT EXISTS counters_tech_key_idx ON counters (tech_key);

CREATE TABLE IF NOT EXISTS features (
    tech_key        TEXT        PRIMARY KEY,
    candidate_id    TEXT,
    query_id        TEXT,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- Справочники и обучающая выборка (100 сигналов + 60 негативов)
-- Expert-колонки датасета (rationale, stage, trend, score, companies) — метаданные,
-- не признаки модели. Признаки считаются только по counters / source_totals.
-- Дата среза: 2026-09-01. Период сбора: 2020-09-01 … 2026-08-31.
-- =============================================================================

CREATE TABLE IF NOT EXISTS areas (
    name            TEXT        PRIMARY KEY,
    name_en         TEXT        NOT NULL,
    sort_order      SMALLINT    NOT NULL,
    description     TEXT        NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS project_settings (
    key             TEXT        PRIMARY KEY,
    value           TEXT        NOT NULL,
    note            TEXT        NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS model_versions (
    version         TEXT        PRIMARY KEY,
    is_default      BOOLEAN     NOT NULL DEFAULT FALSE,
    features        TEXT[]      NOT NULL,
    threshold       NUMERIC     NOT NULL,
    cutoff_date     DATE        NOT NULL,
    intercept       NUMERIC,
    weights         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metrics         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    notes           TEXT        NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS technologies (
    tech_id             TEXT        PRIMARY KEY,
    name                TEXT        NOT NULL,
    name_ru             TEXT        NOT NULL,
    name_en             TEXT,
    name_en_manual      TEXT,
    name_gloss          TEXT,
    name_inline_gloss   TEXT,
    tech_key            TEXT,
    area                TEXT        NOT NULL REFERENCES areas (name),
    label               SMALLINT    NOT NULL,
    source              TEXT        NOT NULL,
    negative_type       TEXT,
    pair_with           INTEGER,
    criterion           TEXT,
    rationale           TEXT,
    stage_raw           TEXT,
    trend_raw           TEXT,
    expert_score        SMALLINT,
    search_terms_manual TEXT,
    terms               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    context_terms       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    duplicate_group     TEXT,
    anchor_volume       INTEGER,
    n_attested_names    INTEGER,
    blind_choice        BOOLEAN,
    naming_model        TEXT,
    prompt_version_norm TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT technologies_label_chk CHECK (label IN (0, 1)),
    CONSTRAINT technologies_source_chk CHECK (source IN ('dataset', 'negatives')),
    CONSTRAINT technologies_negtype_chk CHECK (
        negative_type IS NULL OR negative_type IN ('mature', 'hype', 'fading')
    ),
    CONSTRAINT technologies_id_chk CHECK (tech_id ~ '^(s|n)[0-9]+$'),
    CONSTRAINT technologies_pair_chk CHECK (pair_with IS NULL OR pair_with BETWEEN 1 AND 100),
    CONSTRAINT technologies_score_chk CHECK (expert_score IS NULL OR expert_score BETWEEN 3 AND 7)
);

CREATE INDEX IF NOT EXISTS technologies_tech_key_idx
    ON technologies (tech_key) WHERE tech_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS technologies_area_label_idx ON technologies (area, label);
CREATE INDEX IF NOT EXISTS technologies_group_idx ON technologies (duplicate_group);

CREATE TABLE IF NOT EXISTS technology_companies (
    tech_id         TEXT        NOT NULL REFERENCES technologies (tech_id) ON DELETE CASCADE,
    company         TEXT        NOT NULL,
    position        SMALLINT    NOT NULL DEFAULT 0,
    PRIMARY KEY (tech_id, company)
);

CREATE TABLE IF NOT EXISTS technology_evidence (
    id              BIGSERIAL   PRIMARY KEY,
    tech_id         TEXT        NOT NULL REFERENCES technologies (tech_id) ON DELETE CASCADE,
    title           TEXT,
    url             TEXT        NOT NULL,
    kind            TEXT        NOT NULL,
    position        SMALLINT    NOT NULL DEFAULT 0,
    CONSTRAINT technology_evidence_kind_chk CHECK (kind IN (
        'dataset_source', 'evidence', 'evidence_2'
    )),
    CONSTRAINT technology_evidence_url_chk CHECK (url ~ '^https?://')
);

CREATE INDEX IF NOT EXISTS technology_evidence_tech_idx ON technology_evidence (tech_id);

CREATE TABLE IF NOT EXISTS signal_groups (
    signal_id       INTEGER     PRIMARY KEY,
    group_name      TEXT        NOT NULL,
    reason          TEXT        NOT NULL DEFAULT '',
    CONSTRAINT signal_groups_id_chk CHECK (signal_id BETWEEN 1 AND 100)
);

CREATE TABLE IF NOT EXISTS company_stoplist (
    phrase          TEXT        PRIMARY KEY
);

-- Зафиксированные на дату обучения корпусные итоги. Не смешивать с живым source_totals:
-- корпуса растут, пересчёт сдвигает признаки. Меняются только вместе с моделью.
CREATE TABLE IF NOT EXISTS source_totals_training (
    source              TEXT        NOT NULL,
    period              TEXT        NOT NULL,
    n_total             INTEGER     NOT NULL,
    available           BOOLEAN     NOT NULL DEFAULT TRUE,
    type_filter         TEXT,
    totals_signature    TEXT,
    collected_at        TIMESTAMPTZ,
    PRIMARY KEY (source, period),
    CONSTRAINT source_totals_training_n_chk CHECK (n_total >= 0)
);

CREATE TABLE IF NOT EXISTS rospatent_counts (
    tech_id         TEXT        PRIMARY KEY REFERENCES technologies (tech_id) ON DELETE CASCADE,
    phrase          TEXT        NOT NULL,
    n_pat           INTEGER     NOT NULL,
    fetched_at      TIMESTAMPTZ,
    CONSTRAINT rospatent_counts_n_chk CHECK (n_pat >= 0)
);

CREATE TABLE IF NOT EXISTS eval_labels (
    pool            TEXT        NOT NULL,
    item_key        TEXT        NOT NULL,
    name_en         TEXT,
    label           TEXT,
    category        TEXT,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (pool, item_key)
);

-- Связка технология/кандидат → документ (pipeline.md). documents по-прежнему
-- пишет сборщик с tech_key в строке; эта таблица — каноническая many-to-many.
CREATE TABLE IF NOT EXISTS tech_documents (
    tech_key        TEXT        NOT NULL,
    document_id     BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    role            TEXT        NOT NULL DEFAULT 'search1',
    PRIMARY KEY (tech_key, document_id),
    CONSTRAINT tech_documents_role_chk CHECK (role IN ('search1', 'insight', 'training', 'history'))
);

CREATE INDEX IF NOT EXISTS tech_documents_doc_idx ON tech_documents (document_id);

-- Кэш оценки по ключу технологии и версии модели (pipeline.md, таблица scores).
CREATE TABLE IF NOT EXISTS scores (
    tech_key        TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    query_id        TEXT,
    candidate_id    TEXT,
    score           NUMERIC,
    is_signal       BOOLEAN,
    threshold       NUMERIC,
    contributions   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    top_features    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tech_key, model_version)
);

-- =============================================================================
-- История пользовательских запросов (режим 2)
-- result — снимок контракта для повтора в UI; candidates — нормализованные строки.
-- =============================================================================

CREATE TABLE IF NOT EXISTS queries (
    query_id            TEXT        PRIMARY KEY,
    topic               TEXT        NOT NULL,
    area                TEXT,
    status              TEXT        NOT NULL,
    progress_stage      TEXT,
    progress_done       INTEGER     NOT NULL DEFAULT 0,
    progress_total      INTEGER     NOT NULL DEFAULT 6,
    model_version       TEXT,
    threshold           NUMERIC,
    cutoff_date         DATE,
    extract_version     TEXT,
    naming_mode         TEXT,
    candidates_version  TEXT,
    stats               JSONB       NOT NULL DEFAULT '{}'::jsonb,
    timings             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    warnings            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    error               TEXT,
    result              JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    CONSTRAINT queries_status_chk CHECK (status IN ('queued', 'running', 'done', 'error')),
    CONSTRAINT queries_topic_chk CHECK (length(btrim(topic)) > 0)
);

CREATE INDEX IF NOT EXISTS queries_created_idx ON queries (created_at DESC);
CREATE INDEX IF NOT EXISTS queries_status_idx ON queries (status);
CREATE INDEX IF NOT EXISTS queries_area_idx ON queries (area);

CREATE TABLE IF NOT EXISTS subqueries (
    subquery_id     TEXT        PRIMARY KEY,
    query_id        TEXT        NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    language        TEXT        NOT NULL DEFAULT 'en',
    text            TEXT        NOT NULL,
    position        SMALLINT    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS subqueries_query_idx ON subqueries (query_id);

CREATE TABLE IF NOT EXISTS candidates (
    id                  BIGSERIAL   PRIMARY KEY,
    query_id            TEXT        NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    candidate_id        TEXT,
    name_ru             TEXT,
    name_en             TEXT        NOT NULL,
    tech_key            TEXT,
    bucket              TEXT        NOT NULL,
    rank                INTEGER,
    score               NUMERIC,
    is_signal           BOOLEAN,
    skipped_reason      TEXT,
    reason_ru           TEXT,
    explanation_ru      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    contributions       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    counters            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    features            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    n_pat               INTEGER,
    share_patent        NUMERIC,
    rospatent_failed    BOOLEAN,
    note_ru             TEXT,
    model_version       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT candidates_bucket_chk CHECK (bucket IN ('top', 'excluded')),
    UNIQUE (query_id, name_en)
);

CREATE INDEX IF NOT EXISTS candidates_query_idx ON candidates (query_id, bucket, rank);
CREATE INDEX IF NOT EXISTS candidates_tech_key_idx ON candidates (tech_key);

CREATE TABLE IF NOT EXISTS candidate_sources (
    id              BIGSERIAL   PRIMARY KEY,
    candidate_pk    BIGINT      NOT NULL REFERENCES candidates (id) ON DELETE CASCADE,
    title           TEXT,
    url             TEXT,
    published_at    DATE,
    source          TEXT,
    source_type     TEXT,
    language        TEXT,
    trust_level     TEXT
);

CREATE INDEX IF NOT EXISTS candidate_sources_cand_idx ON candidate_sources (candidate_pk);

CREATE TABLE IF NOT EXISTS insights (
    query_id        TEXT        NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    rank            INTEGER     NOT NULL,
    candidate_pk    BIGINT      REFERENCES candidates (id) ON DELETE SET NULL,
    status          TEXT        NOT NULL DEFAULT 'pending',
    description_ru  TEXT,
    advantages_ru   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    cases_ru        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    generated_at    TIMESTAMPTZ,
    PRIMARY KEY (query_id, rank),
    CONSTRAINT insights_status_chk CHECK (status IN ('pending', 'done', 'error'))
);

CREATE OR REPLACE VIEW v_training_balance AS
SELECT
    area,
    count(*) FILTER (WHERE label = 1) AS signals,
    count(*) FILTER (WHERE label = 0) AS negatives,
    count(*) AS total
FROM technologies
GROUP BY area;

CREATE OR REPLACE VIEW v_query_history AS
SELECT
    q.query_id,
    q.topic,
    q.area,
    q.status,
    q.model_version,
    q.created_at,
    q.finished_at,
    q.stats ->> 'candidates_found' AS candidates_found,
    q.stats ->> 'above_075' AS above_075,
    q.stats ->> 'documents_total' AS documents_total
FROM queries q;

CREATE OR REPLACE VIEW v_technologies AS
SELECT
    t.tech_id,
    t.name,
    t.name_ru,
    t.name_en,
    t.tech_key,
    t.area,
    t.label,
    t.source,
    t.negative_type,
    t.pair_with,
    t.duplicate_group,
    t.expert_score,
    t.stage_raw,
    t.criterion,
    t.rationale,
    r.n_pat,
    r.phrase AS patent_phrase
FROM technologies t
LEFT JOIN rospatent_counts r ON r.tech_id = t.tech_id;
