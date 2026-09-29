"""Инсайт по клику: формат API читается UI (ui.view.insight_view), повторный запрос отдаёт сохранённый."""

from __future__ import annotations

from ui import view


def test_insight_is_generated_from_documents(client, finished) -> None:
    query_id = finished()["query_id"]
    rank = client.get(f"/queries/{query_id}").json()["top"][0]["rank"]
    insight = client.get(f"/queries/{query_id}/insights/{rank}").json()
    shown = view.insight_view(insight)
    assert shown is not None and shown["description"] and shown["sources"]
    again = client.get(f"/queries/{query_id}/insights/{rank}").json()
    assert again["description_ru"] == insight["description_ru"]


def test_insight_built_by_pipeline_is_served_without_llm(client, finished, monkeypatch) -> None:
    """Инсайт, построенный run_query до выдачи, отдаётся как есть: build_insight по клику не вызывается."""
    import pipeline.run_query as rq

    fake_run_query = rq.run_query  # уже подменён в conftest готовым прогоном

    def with_insight(topic, area=None, **kwargs):
        result = fake_run_query(topic, area, **kwargs)
        result["top"][0]["insight"] = {"status": "done", "description_ru": "готовое описание",
                                       "advantages_ru": [], "cases_ru": [], "sources": []}
        return result

    def no_build(*args, **kwargs):
        raise AssertionError("build_insight вызван по клику")

    monkeypatch.setattr(rq, "run_query", with_insight)
    monkeypatch.setattr("pipeline.insights.build_insight", no_build)
    query_id = finished()["query_id"]
    insight = client.get(f"/queries/{query_id}/insights/1").json()
    assert insight["description_ru"] == "готовое описание" and insight["rank"] == 1


def test_insight_for_unknown_rank_is_404(client, finished) -> None:
    query_id = finished()["query_id"]
    assert client.get(f"/queries/{query_id}/insights/99").status_code == 404
