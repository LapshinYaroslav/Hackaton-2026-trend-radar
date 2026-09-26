"""Идемпотентный посев справочников и обучающей выборки."""

from __future__ import annotations

import json
from typing import Any

from db.loaders import (
    AREAS,
    MODEL_VERSIONS,
    PROJECT_SETTINGS,
    load_eval_labels,
    load_negatives,
    load_rospatent_counts,
    load_signal_groups,
    load_signals_xlsx,
    load_source_totals_training,
    load_stoplist,
    load_technologies_csv,
    merge_training,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def seed(conn) -> dict[str, int]:
    """Заливает справочники. Повторный запуск обновляет строки, не плодит дубли."""
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        counts["areas"] = _seed_areas(cur)
        counts["settings"] = _seed_settings(cur)
        counts["models"] = _seed_models(cur)
        training = merge_training(
            load_technologies_csv(),
            load_negatives(),
            load_signal_groups(),
            load_signals_xlsx(),
        )
        counts["technologies"] = _seed_technologies(cur, training)
        counts["companies"] = _seed_companies(cur, training)
        counts["evidence"] = _seed_evidence(cur, training)
        counts["groups"] = _seed_groups(cur, load_signal_groups())
        counts["stoplist"] = _seed_stoplist(cur, load_stoplist())
        counts["source_totals_training"] = _seed_source_totals(cur, load_source_totals_training())
        counts["rospatent"] = _seed_rospatent(cur, load_rospatent_counts())
        counts["eval_labels"] = _seed_eval(cur, load_eval_labels())
    conn.commit()
    return counts


def _seed_areas(cur) -> int:
    for name, name_en, sort_order, description in AREAS:
        cur.execute(
            """
            INSERT INTO areas (name, name_en, sort_order, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                name_en = EXCLUDED.name_en,
                sort_order = EXCLUDED.sort_order,
                description = EXCLUDED.description
            """,
            (name, name_en, sort_order, description),
        )
    return len(AREAS)


def _seed_settings(cur) -> int:
    for key, value, note in PROJECT_SETTINGS:
        cur.execute(
            """
            INSERT INTO project_settings (key, value, note)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, note = EXCLUDED.note
            """,
            (key, value, note),
        )
    return len(PROJECT_SETTINGS)


def _seed_models(cur) -> int:
    for row in MODEL_VERSIONS:
        cur.execute(
            """
            INSERT INTO model_versions
                (version, is_default, features, threshold, cutoff_date, intercept, weights, metrics, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (version) DO UPDATE SET
                is_default = EXCLUDED.is_default,
                features = EXCLUDED.features,
                threshold = EXCLUDED.threshold,
                cutoff_date = EXCLUDED.cutoff_date,
                intercept = EXCLUDED.intercept,
                weights = EXCLUDED.weights,
                metrics = EXCLUDED.metrics,
                notes = EXCLUDED.notes
            """,
            (
                row["version"],
                row["is_default"],
                row["features"],
                row["threshold"],
                row["cutoff_date"],
                row["intercept"],
                _json(row["weights"]),
                _json(row["metrics"]),
                row["notes"],
            ),
        )
    return len(MODEL_VERSIONS)


def _seed_technologies(cur, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        cur.execute(
            """
            INSERT INTO technologies (
                tech_id, name, name_ru, name_en, name_en_manual, name_gloss, name_inline_gloss,
                tech_key, area, label, source, negative_type, pair_with, criterion,
                rationale, stage_raw, trend_raw, expert_score, search_terms_manual,
                terms, context_terms, duplicate_group, anchor_volume, n_attested_names,
                blind_choice, naming_model, prompt_version_norm, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s, %s, %s,
                %s, %s, %s, now()
            )
            ON CONFLICT (tech_id) DO UPDATE SET
                name = EXCLUDED.name,
                name_ru = EXCLUDED.name_ru,
                name_en = EXCLUDED.name_en,
                name_en_manual = EXCLUDED.name_en_manual,
                name_gloss = EXCLUDED.name_gloss,
                name_inline_gloss = EXCLUDED.name_inline_gloss,
                tech_key = EXCLUDED.tech_key,
                area = EXCLUDED.area,
                label = EXCLUDED.label,
                source = EXCLUDED.source,
                negative_type = EXCLUDED.negative_type,
                pair_with = EXCLUDED.pair_with,
                criterion = EXCLUDED.criterion,
                rationale = COALESCE(EXCLUDED.rationale, technologies.rationale),
                stage_raw = COALESCE(EXCLUDED.stage_raw, technologies.stage_raw),
                trend_raw = COALESCE(EXCLUDED.trend_raw, technologies.trend_raw),
                expert_score = COALESCE(EXCLUDED.expert_score, technologies.expert_score),
                search_terms_manual = EXCLUDED.search_terms_manual,
                terms = EXCLUDED.terms,
                context_terms = EXCLUDED.context_terms,
                duplicate_group = EXCLUDED.duplicate_group,
                anchor_volume = EXCLUDED.anchor_volume,
                n_attested_names = EXCLUDED.n_attested_names,
                blind_choice = EXCLUDED.blind_choice,
                naming_model = EXCLUDED.naming_model,
                prompt_version_norm = EXCLUDED.prompt_version_norm,
                updated_at = now()
            """,
            (
                row["tech_id"],
                row["name"],
                row["name_ru"],
                row.get("name_en"),
                row.get("name_en_manual"),
                row.get("name_gloss"),
                row.get("name_inline_gloss"),
                row.get("tech_key"),
                row["area"],
                row["label"],
                row["source"],
                row.get("negative_type") or None,
                row.get("pair_with"),
                row.get("criterion"),
                row.get("rationale"),
                row.get("stage_raw"),
                row.get("trend_raw"),
                row.get("expert_score"),
                row.get("search_terms_manual"),
                _json(row.get("terms") or []),
                _json(row.get("context_terms") or []),
                row.get("duplicate_group"),
                row.get("anchor_volume"),
                row.get("n_attested_names"),
                row.get("blind_choice"),
                row.get("naming_model"),
                row.get("prompt_version_norm"),
            ),
        )
    return len(rows)


def _seed_companies(cur, rows: list[dict[str, Any]]) -> int:
    written = 0
    for row in rows:
        companies = row.get("companies") or []
        if not companies:
            continue
        cur.execute("DELETE FROM technology_companies WHERE tech_id = %s", (row["tech_id"],))
        for position, company in enumerate(companies):
            cur.execute(
                """
                INSERT INTO technology_companies (tech_id, company, position)
                VALUES (%s, %s, %s)
                ON CONFLICT (tech_id, company) DO UPDATE SET position = EXCLUDED.position
                """,
                (row["tech_id"], company, position),
            )
            written += 1
    return written


def _seed_evidence(cur, rows: list[dict[str, Any]]) -> int:
    written = 0
    for row in rows:
        evidence = row.get("evidence") or []
        if not evidence:
            continue
        cur.execute("DELETE FROM technology_evidence WHERE tech_id = %s", (row["tech_id"],))
        for position, item in enumerate(evidence):
            cur.execute(
                """
                INSERT INTO technology_evidence (tech_id, title, url, kind, position)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (row["tech_id"], item.get("title"), item["url"], item["kind"], position),
            )
            written += 1
    return written


def _seed_groups(cur, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        cur.execute(
            """
            INSERT INTO signal_groups (signal_id, group_name, reason)
            VALUES (%s, %s, %s)
            ON CONFLICT (signal_id) DO UPDATE SET
                group_name = EXCLUDED.group_name,
                reason = EXCLUDED.reason
            """,
            (row["signal_id"], row["group_name"], row["reason"]),
        )
    return len(rows)


def _seed_stoplist(cur, phrases: list[str]) -> int:
    for phrase in phrases:
        cur.execute(
            "INSERT INTO company_stoplist (phrase) VALUES (%s) ON CONFLICT (phrase) DO NOTHING",
            (phrase,),
        )
    return len(phrases)


def _seed_source_totals(cur, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        cur.execute(
            """
            INSERT INTO source_totals_training
                (source, period, n_total, available, type_filter, totals_signature, collected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, period) DO UPDATE SET
                n_total = EXCLUDED.n_total,
                available = EXCLUDED.available,
                type_filter = EXCLUDED.type_filter,
                totals_signature = EXCLUDED.totals_signature,
                collected_at = EXCLUDED.collected_at
            """,
            (
                row["source"],
                row["period"],
                row["n_total"],
                row["available"],
                row.get("type_filter"),
                row.get("totals_signature"),
                row.get("collected_at"),
            ),
        )
    return len(rows)


def _seed_rospatent(cur, rows: list[dict[str, Any]]) -> int:
    written = 0
    for row in rows:
        cur.execute("SELECT 1 FROM technologies WHERE tech_id = %s", (row["tech_id"],))
        if cur.fetchone() is None:
            continue
        cur.execute(
            """
            INSERT INTO rospatent_counts (tech_id, phrase, n_pat, fetched_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tech_id) DO UPDATE SET
                phrase = EXCLUDED.phrase,
                n_pat = EXCLUDED.n_pat,
                fetched_at = EXCLUDED.fetched_at
            """,
            (row["tech_id"], row["phrase"], row["n_pat"], row.get("fetched_at")),
        )
        written += 1
    return written


def _seed_eval(cur, rows: list[dict[str, Any]]) -> int:
    for row in rows:
        cur.execute(
            """
            INSERT INTO eval_labels (pool, item_key, name_en, label, category, payload)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (pool, item_key) DO UPDATE SET
                name_en = EXCLUDED.name_en,
                label = EXCLUDED.label,
                category = EXCLUDED.category,
                payload = EXCLUDED.payload
            """,
            (
                row["pool"],
                row["item_key"],
                row.get("name_en"),
                row.get("label"),
                row.get("category"),
                _json(row.get("payload") or {}),
            ),
        )
    return len(rows)
