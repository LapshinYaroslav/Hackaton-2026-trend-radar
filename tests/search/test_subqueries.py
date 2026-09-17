"""Тесты генерации подзапросов: ask_llm подменяется, реальных запросов нет."""
import json
import threading
import unittest
from unittest.mock import patch

from search import subqueries as sq

MODEL_URI = "gpt://test/yandexgpt/latest"
RU = ["фотонные вычисления", "оптический интерконнект", "мемристорные матрицы",
      "нейроморфные ускорители", "квантовые сенсоры"]
EN = ["photonic computing", "optical interconnect", "memristor crossbar arrays",
      "neuromorphic accelerators", "quantum sensing"]


def answer(text: str | None, error: str | None = None) -> dict:
    """Ответ в формате ask_llm."""
    return {"text": text, "model_uri": MODEL_URI, "model_version": "test",
            "usage": {}, "elapsed_s": 0.1, "error": error}


def payload(language: str, items: list) -> str:
    """JSON-ответ модели на один язык."""
    return json.dumps({language: items}, ensure_ascii=False)


def texts(result: dict, language: str) -> list[str]:
    """Тексты подзапросов одного языка."""
    return [item["text"] for item in result["subqueries"] if item["language"] == language]


def language_of(system_prompt: str) -> str:
    """Какой язык просит системный промпт."""
    return "ru" if '"ru"' in system_prompt else "en"


class GenerateSubqueriesTest(unittest.TestCase):
    def _generate(self, answers: dict, topic: str = "оптические технологии в дата-центрах", query_id: str = "q7"):
        """Подменяет ask_llm очередями ответов по языкам (вызовы идут из разных потоков)."""
        lock = threading.Lock()
        queues = {lang: list(items) for lang, items in answers.items()}

        def fake_ask_llm(system_prompt, user_prompt, **kwargs):
            with lock:
                queue = queues.get(language_of(system_prompt)) or []
                return queue.pop(0) if queue else answer(None, "ответы кончились")

        with patch.object(sq, "ask_llm", side_effect=fake_ask_llm) as mock:
            return sq.generate_subqueries(topic, query_id), mock

    def test_normal_answer(self):
        """По одному вызову на язык, 5 и 5, правильные идентификаторы, без предупреждений."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(len(result["subqueries"]), 10)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["model_uri"], MODEL_URI)
        self.assertEqual(result["subqueries"][0]["subquery_id"], "q7-ru-1")
        self.assertEqual(result["subqueries"][5]["subquery_id"], "q7-en-1")
        self.assertEqual(texts(result, "ru"), RU)

    def test_json_fence(self):
        """Обёртка ```json снимается."""
        result, _ = self._generate({"ru": [answer("```json\n" + payload("ru", RU) + "\n```")],
                                    "en": [answer("```\n" + payload("en", EN) + "\n```")]})
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(texts(result, "en"), EN)

    def test_wrong_language_dropped(self):
        """Английская строка в ru и кириллица в en отбрасываются."""
        result, _ = self._generate({"ru": [answer(payload("ru", RU + ["optical computing"]))],
                                    "en": [answer(payload("en", EN + ["фотоника"]))]})
        self.assertNotIn("optical computing", texts(result, "ru"))
        self.assertNotIn("фотоника", texts(result, "en"))

    def test_duplicates_and_meta_words_dropped(self):
        """Дубли по регистру и мета-слова не попадают в результат."""
        noisy_ru = [RU[0], RU[0].upper(), "тренды фотоники", "перспективные материалы"] + RU[1:]
        noisy_en = [EN[0], EN[0].title(), "emerging photonics"] + EN[1:]
        result, _ = self._generate({"ru": [answer(payload("ru", noisy_ru))],
                                    "en": [answer(payload("en", noisy_en))]})
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(texts(result, "en"), EN)

    def test_no_more_than_limit(self):
        """Лишние подзапросы обрезаются до N."""
        result, _ = self._generate({"ru": [answer(payload("ru", RU + ["лишний подзапрос"]))],
                                    "en": [answer(payload("en", EN + ["extra subquery"]))]})
        self.assertEqual(len(texts(result, "ru")), sq.N_SUBQUERIES_RU)
        self.assertEqual(len(texts(result, "en")), sq.N_SUBQUERIES_EN)

    def test_retry_only_short_language(self):
        """Не хватает русских: повтор идёт только по русскому, английский не перезапрашивается."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU[:2])), answer(payload("ru", RU[1:]))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(texts(result, "en"), EN)
        self.assertIn("подзапросов меньше нужного (ru), выполнен повтор", result["warnings"])

    def test_still_not_enough_after_retry(self):
        """После повтора всё равно мало: возвращаем сколько есть с предупреждением."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU[:2])), answer(payload("ru", RU[:2]))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(len(texts(result, "ru")), 2)
        self.assertIn("подзапросов на языке ru: 2 из 5", result["warnings"])

    def test_broken_json_and_network_error(self):
        """Битый JSON и ошибка сети: пустой результат и понятные предупреждения."""
        result, _ = self._generate({"ru": [answer("не json"), answer("тоже не json")],
                                    "en": [answer(None, "ConnectionError: нет сети"),
                                           answer(None, "ConnectionError: нет сети")]})
        self.assertEqual(result["subqueries"], [])
        self.assertTrue(any("разбор ответа (ru) не удался" in w for w in result["warnings"]))
        self.assertTrue(any("вызов LLM (en) не удался" in w for w in result["warnings"]))

    def test_plan_field_ignored(self):
        """Поле plan из ответа модели не попадает в подзапросы."""
        with_plan = json.dumps({"plan": ["фотоника", "память"], "ru": RU}, ensure_ascii=False)
        result, _ = self._generate({"ru": [answer(with_plan)], "en": [answer(payload("en", EN))]})
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(len(result["subqueries"]), 10)

    def test_generalization_dropped(self):
        """Подзапрос из одних только слов темы отбрасывается как обобщение."""
        good = ["квантовый метод монте-карло", "вариационная оптимизация портфеля",
                "постквантовые подписи платежей", "квантовый отжиг маршрутизации", "амплитудная оценка риска"]
        result, _ = self._generate(
            {"ru": [answer(payload("ru", ["квантовые технологии", "квантовые финансы"] + good))],
             "en": [answer(payload("en", EN))]},
            topic="квантовые технологии в финансах")
        self.assertNotIn("квантовые технологии", texts(result, "ru"))
        self.assertNotIn("квантовые финансы", texts(result, "ru"))
        self.assertEqual(texts(result, "ru"), good)

    def test_empty_topic(self):
        """Пустая тема: ошибка до обращения к LLM."""
        with patch.object(sq, "ask_llm") as mock:
            with self.assertRaises(ValueError):
                sq.generate_subqueries("   ", "q1")
        self.assertEqual(mock.call_count, 0)


class GenerateManyTest(unittest.TestCase):
    def test_batch_keeps_order_and_ids(self):
        """Пакет тем: порядок сохраняется, идентификаторы по умолчанию q1, q2."""
        lock = threading.Lock()
        calls = []

        def fake_ask_llm(system_prompt, user_prompt, **kwargs):
            language = language_of(system_prompt)
            with lock:
                calls.append(language)
            return answer(payload(language, RU if language == "ru" else EN))

        with patch.object(sq, "ask_llm", side_effect=fake_ask_llm):
            results = sq.generate_subqueries_many(["фотоника", "мемристоры"])
        self.assertEqual([item["topic"] for item in results], ["фотоника", "мемристоры"])
        self.assertEqual([item["query_id"] for item in results], ["q1", "q2"])
        self.assertEqual(len(calls), 4)

    def test_empty_batch(self):
        """Пустой список тем не вызывает LLM."""
        with patch.object(sq, "ask_llm") as mock:
            self.assertEqual(sq.generate_subqueries_many([]), [])
        self.assertEqual(mock.call_count, 0)

    def test_ids_length_mismatch(self):
        """Идентификаторов меньше, чем тем: ошибка до вызовов."""
        with patch.object(sq, "ask_llm") as mock:
            with self.assertRaises(ValueError):
                sq.generate_subqueries_many(["а", "б"], ["q1"])
        self.assertEqual(mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
