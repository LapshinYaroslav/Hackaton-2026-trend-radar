from pipeline.weak_sources import WEAK_ONLY_REASON, is_weak_only, move_weak_only


def test_blog_and_press_alone_are_weak() -> None:
    item = {
        "name_en": "foo bar",
        "sources": [
            {"source_type": "blog", "url": "https://a.example/1"},
            {"source_type": "press_release", "url": "https://a.example/2"},
        ],
    }
    assert is_weak_only(item)


def test_paper_keeps_signal() -> None:
    item = {
        "name_en": "foo bar",
        "sources": [
            {"source_type": "blog", "url": "https://a.example/1"},
            {"source_type": "paper", "url": "https://a.example/2"},
        ],
    }
    assert not is_weak_only(item)


def test_move_weak_only_reranks() -> None:
    top = [
        {
            "rank": 1,
            "name_en": "weak tech",
            "name_ru": "слабая",
            "sources": [{"source_type": "blog", "url": "https://a.example/1"}],
        },
        {
            "rank": 2,
            "name_en": "real tech",
            "name_ru": "настоящая",
            "sources": [{"source_type": "preprint", "url": "https://a.example/2"}],
        },
    ]
    kept, excluded = move_weak_only(top, [])
    assert [item["name_en"] for item in kept] == ["real tech"]
    assert kept[0]["rank"] == 1
    assert excluded[0]["skipped_reason"] == WEAK_ONLY_REASON
