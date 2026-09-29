import joblib
import pytest

from model.bootstrap import ensure_artifact


def _artifact_dir(tmp_path, monkeypatch):
    """Каталог артефактов во временной папке."""
    from model import train as train_mod

    monkeypatch.setattr(train_mod, "ARTIFACT_DIR", tmp_path / "model")
    return tmp_path / "model" / "s2a2-v1"


def test_missing_artifact_raises(tmp_path, monkeypatch) -> None:
    _artifact_dir(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Нет обученной модели s2a2-v1"):
        ensure_artifact("s2a2-v1")


def test_bootstrap_stub_raises(tmp_path, monkeypatch) -> None:
    folder = _artifact_dir(tmp_path, monkeypatch)
    folder.mkdir(parents=True)
    joblib.dump({"meta": {"bootstrap": True}, "pipeline": None}, folder / "trend_radar_s2a2-v1.joblib")
    with pytest.raises(RuntimeError, match="заглушка"):
        ensure_artifact("s2a2-v1")


def test_trained_artifact_passes(tmp_path, monkeypatch) -> None:
    folder = _artifact_dir(tmp_path, monkeypatch)
    folder.mkdir(parents=True)
    path = folder / "trend_radar_s2a2-v1.joblib"
    joblib.dump({"meta": {"model_version": "s2a2-v1"}, "pipeline": None}, path)
    assert ensure_artifact("s2a2-v1") == path
