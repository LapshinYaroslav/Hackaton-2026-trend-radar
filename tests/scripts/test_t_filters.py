"""Тесты фильтров задачи Т: определения из условия и граничные случаи."""
import numpy as np
import pytest

from scripts.t_filters import (are_duplicates, duplicate_groups, is_mature, is_template,
                               mature_threshold, phrase_key, repeat_counts, title_key,
                               unique_documents)


def docs(*pairs: tuple[str, str]) -> list[dict]:
    """Склеенные документы из пар (заголовок, текст)."""
    return unique_documents([{"title": title, "text": text} for title, text in pairs])[0]


def test_mature_threshold_is_numpy_linear_percentile():
    values = [1, 5, 7, 20, 100, 3]
    assert mature_threshold(values) == pytest.approx(np.percentile(values, 95))


def test_mature_is_strictly_greater():
    assert is_mature(101, 100.0) and not is_mature(100, 100.0)


def test_phrase_key_stems_and_splits():
    assert phrase_key("Robotic-Arms/grippers") == ("robot", "arm", "gripper")
    assert phrase_key("robots") == phrase_key("robotic") == ("robot",)
    assert phrase_key("") == ()


def test_duplicates_equal_and_contained():
    assert are_duplicates(phrase_key("humanoid robots"), phrase_key("humanoid robotic"))
    assert are_duplicates(phrase_key("humanoid robot"), phrase_key("bipedal humanoid robot safety"))
    assert not are_duplicates(phrase_key("humanoid robot"), phrase_key("humanoid for robot"))


def test_single_word_key_is_not_matched_by_containment():
    assert not are_duplicates(phrase_key("robot"), phrase_key("robot learning"))
    assert are_duplicates(phrase_key("robots"), phrase_key("robot"))
    assert not are_duplicates((), ())


def test_groups_are_transitive_and_led_by_max_score():
    terms = ["robot learning policy", "robot learning", "learning policy", "drone swarm"]
    groups = duplicate_groups(terms, [0.5, 0.9, 0.7, 0.8])
    # «robot learning» и «learning policy» друг другу не дубли, но оба входят в первую фразу
    assert sorted(groups) == [[1, 2, 0], [3]]


def test_groups_tie_keeps_first():
    assert duplicate_groups(["soft gripper", "soft grippers"], [0.6, 0.6]) == [[0, 1]]
    assert duplicate_groups([], []) == []


@pytest.mark.parametrize("term, expected", [
    ("AI-driven inspection system", True), ("smart manufacturing platform", True),
    ("novel grasping method", True), ("smart system", True), ("system smart", False),
    ("inspection system", False), ("smart grippers", False), ("ai driven system", False),
    ("", False)])
def test_template(term, expected):
    assert is_template(term) is expected


def test_title_key_drops_punctuation():
    assert title_key("  Soft Robots: A Survey!") == title_key("soft robots a survey")
    assert title_key(None) == ""


def test_unique_documents_merges_by_title_and_counts():
    merged, glued = unique_documents([{"title": "Soft Robots: A Survey", "text": "a"},
                                      {"title": "soft robots - a survey", "text": "b"},
                                      {"title": "", "text": "c"}, {"title": "", "text": "d"}])
    assert glued == 1 and len(merged) == 3
    assert merged[0]["text"] == "a b"


def test_repeat_counts_plural_and_word_boundary():
    pool = docs(("Humanoid robots at work", ""), ("", "a humanoid-robot demo"),
                ("Robotics news", "humanoid robotics is not a match"))
    assert repeat_counts("humanoid robot", pool) == (2, 2)
    assert repeat_counts("box", docs(("boxes", ""))) == (1, 1)


def test_repeat_counts_merged_title_counts_once():
    pool = docs(("Soft grippers", "soft gripper"), ("Soft grippers.", "soft gripper"))
    assert repeat_counts("soft gripper", pool) == (1, 1)


def test_repeat_counts_empty():
    assert repeat_counts("", docs(("x", "y"))) == (0, 0)
    assert repeat_counts("robot", []) == (0, 0)


def test_authors_are_counted_when_present():
    pool = unique_documents([{"title": "a", "text": "soft gripper", "authors": ["Li"]},
                             {"title": "b", "text": "soft gripper", "authors": ["Li", "Wu"]},
                             {"title": "c", "text": "soft gripper"}])[0]
    assert repeat_counts("soft gripper", pool) == (3, 2)
