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


def test_insight_for_unknown_rank_is_404(client, finished) -> None:
    query_id = finished()["query_id"]
    assert client.get(f"/queries/{query_id}/insights/99").status_code == 404
