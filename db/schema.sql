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

CREATE TABLE IF NOT EXISTS source_totals (
    source          TEXT        NOT NULL,
    window          TEXT        NOT NULL,
    n_total         INTEGER     NOT NULL,
    available       BOOLEAN     NOT NULL DEFAULT TRUE,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, window),
    CONSTRAINT source_totals_window_chk CHECK (window IN ('before', 'now'))
);

CREATE TABLE IF NOT EXISTS features (
    tech_key        TEXT        PRIMARY KEY,
    candidate_id    TEXT,
    query_id        TEXT,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
