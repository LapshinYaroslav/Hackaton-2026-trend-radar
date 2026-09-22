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
