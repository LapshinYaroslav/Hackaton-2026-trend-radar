from model.bootstrap import ensure_artifact, write_published
from model.predict import load, predict


def test_published_artifact_scores(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from model import train as train_mod
    from model import bootstrap as boot

    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", tmp_path / "data" / "model")
    monkeypatch.setattr(boot, "try_train", lambda version="s2a2-v1": None)
    path = write_published("s2a2-v1")
    assert path.exists()
    artifact = load(path)
    assert artifact["meta"]["model_version"] == "s2a2-v1"
    score, parts = predict(
        {
            "share_news_wordmatch": 0.8,
            "recency": 0.9,
            "share_patent": 0.0,
            "age_first_arxiv": 1.0,
            "share_prev6": 0.0,
            "growth_research": 0.1,
        },
        area="Роботы",
        artifact=artifact,
    )
    assert 0 < score < 1
    assert "share_news_wordmatch" in parts


def test_ensure_artifact_reuses_existing(tmp_path, monkeypatch) -> None:
    from model import train as train_mod

    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", tmp_path / "model")
    first = write_published("s2a2-v1")
    again = ensure_artifact("s2a2-v1")
    assert again == first or again.exists()
