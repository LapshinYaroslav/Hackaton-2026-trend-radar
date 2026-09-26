"""Тесты генерации подзапросов: ask_llm подменяется, реальных запросов нет."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from search import subqueries as sq

MODEL_URI = "gpt://test/yandexgpt/latest"
RU = ["фотонные вычисления", "оптический интерконнект", "мемристорные матрицы",
      "нейроморфные ускорители", "квантовые сенсоры"]
EN = ["photonic computing", "optical interconnect", "memristor crossbar arrays",
      "neuromorphic accelerators", "quantum sensing", "spiking neural networks",
      "silicon photonic modulators", "analog inference chips"]


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


class IsolatedTest(unittest.TestCase):
    """Кэш во временной папке, URI модели — заглушка."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache_dir = Path(folder.name)
        for name, value in [("CACHE_DIR", self.cache_dir), ("build_model_uri", lambda: MODEL_URI)]:
            patcher = patch.object(sq, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class GenerateSubqueriesTest(IsolatedTest):
    def _generate(self, answers: dict, topic: str = "оптические технологии в дата-центрах",
                  query_id: str = "q7", use_cache: bool = True):
        """Подменяет ask_llm очередями ответов по языкам (вызовы идут из разных потоков)."""
        lock = threading.Lock()
        queues = {lang: list(items) for lang, items in answers.items()}

        def fake_ask_llm(system_prompt, user_prompt, **kwargs):
            with lock:
                queue = queues.get(language_of(system_prompt)) or []
                return queue.pop(0) if queue else answer(None, "ответы кончились")

        with patch.object(sq, "ask_llm", side_effect=fake_ask_llm) as mock:
            return sq.generate_subqueries(topic, query_id, use_cache=use_cache), mock

    def test_normal_answer(self):
        """По одному вызову на язык, 5 ru и 8 en, идентификаторы, версии, без предупреждений."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(len(result["subqueries"]), 13)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["model_uri"], MODEL_URI)
        self.assertEqual(result["model_version"], "test")
        self.assertEqual(result["prompt_version"], "subq-v3")
        self.assertEqual(result["subqueries"][0]["subquery_id"], "q7-ru-1")
        self.assertEqual(result["subqueries"][5]["subquery_id"], "q7-en-1")
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(texts(result, "en"), EN)

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

    def test_rejected_reason_in_warnings(self):
        """У каждого отброшенного подзапроса причина в warnings."""
        result, _ = self._generate({"ru": [answer(payload("ru", RU))],
                                    "en": [answer(payload("en", EN + ["quantum sensing 2025"]))]})
        self.assertIn("отброшен (en): «quantum sensing 2025» — год", result["warnings"])

    def test_retry_only_short_language(self):
        """Не хватает русских: повтор идёт только по русскому, английский не перезапрашивается."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU[:2])), answer(payload("ru", RU[1:]))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(texts(result, "en"), EN)
        self.assertIn("подзапросов меньше нужного (ru), выполнен повтор", result["warnings"])

    def test_retry_changes_temperature_and_prompt(self):
        """Повтор: температура 0.8 и строка с отброшенными подзапросами и правилом."""
        short_en = EN[:4] + ["accuracy metrics design", "quantum sensing 2025"]
        _, mock = self._generate({"ru": [answer(payload("ru", RU))],
                                  "en": [answer(payload("en", short_en)), answer(payload("en", EN[4:]))]})
        en_calls = [call for call in mock.call_args_list if language_of(call.args[0]) == "en"]
        self.assertEqual([call.kwargs["temperature"] for call in en_calls], [0.3, 0.8])
        retry_prompt = en_calls[1].args[0]
        self.assertTrue(retry_prompt.startswith(en_calls[0].args[0]))
        self.assertIn("«accuracy metrics design» — слово зрелой области metrics", retry_prompt)
        self.assertIn("«quantum sensing 2025» — год", retry_prompt)

    def test_still_not_enough_after_retry(self):
        """После повтора всё равно мало: возвращаем сколько есть с предупреждением."""
        result, mock = self._generate({"ru": [answer(payload("ru", RU[:2])), answer(payload("ru", RU[:2]))],
                                       "en": [answer(payload("en", EN))]})
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(len(texts(result, "ru")), 2)
        self.assertIn("подзапросов на языке ru: 2 из 5", result["warnings"])

    def test_no_english_raises(self):
        """Английских нет после обоих кругов: ValueError с перечнем нарушений."""
        with self.assertRaises(ValueError) as caught:
            self._generate({"ru": [answer(payload("ru", RU))],
                            "en": [answer(None, "ConnectionError: нет сети"),
                                   answer(None, "ConnectionError: нет сети")]})
        self.assertIn("английского", str(caught.exception))
        self.assertIn("вызов LLM (en) не удался", str(caught.exception))

    def test_no_russian_is_warning(self):
        """Русских нет после обоих кругов: запрос идёт дальше с одним английским."""
        result, _ = self._generate({"ru": [answer("не json"), answer("тоже не json")],
                                    "en": [answer(payload("en", EN))]})
        self.assertEqual(texts(result, "ru"), [])
        self.assertEqual(texts(result, "en"), EN)
        self.assertIn("подзапросов на языке ru: 0 из 5", result["warnings"])
        self.assertTrue(any("разбор ответа (ru) не удался" in w for w in result["warnings"]))

    def test_plan_field_ignored(self):
        """Поле plan из ответа модели не попадает в подзапросы."""
        with_plan = json.dumps({"plan": ["фотоника", "память"], "ru": RU}, ensure_ascii=False)
        result, _ = self._generate({"ru": [answer(with_plan)], "en": [answer(payload("en", EN))]})
        self.assertEqual(texts(result, "ru"), RU)
        self.assertEqual(len(result["subqueries"]), 13)

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

    def test_missing_model_version_goes_with_warning(self):
        """API не вернул modelVersion: None в выходе только вместе с warnings, и так же из кэша."""
        no_version = {**answer(payload("en", EN)), "model_version": None}
        ru_no_version = {**answer(payload("ru", RU)), "model_version": None}
        first, _ = self._generate({"ru": [ru_no_version], "en": [no_version]}, topic="мемристоры")
        cached, mock = self._generate({}, topic="мемристоры", query_id="q8")
        for result in (first, cached):
            self.assertIsNone(result["model_version"])
            self.assertIn(sq.MISSING_VERSION, result["warnings"])
        self.assertEqual(mock.call_count, 0)

    def test_cache_second_call_skips_llm(self):
        """Та же тема второй раз: из кэша, без LLM, с новым query_id. use_cache=False зовёт LLM."""
        answers = {"ru": [answer(payload("ru", RU))] * 2, "en": [answer(payload("en", EN))] * 2}
        first, _ = self._generate(answers, topic="Оптика  в ЦОД", query_id="q1")
        second, mock = self._generate(answers, topic="оптика в цод", query_id="q2")
        self.assertEqual(mock.call_count, 0)
        self.assertEqual(texts(second, "en"), texts(first, "en"))
        self.assertEqual(second["subqueries"][0]["subquery_id"], "q2-ru-1")
        _, fresh = self._generate(answers, topic="оптика в цод", use_cache=False)
        self.assertEqual(fresh.call_count, 2)


class CheckSubqueriesTest(IsolatedTest):
    """По одному случаю на каждое правило валидации."""

    def reason(self, text: str, language: str = "en", kept: list | None = None) -> str | None:
        """Причина отказа одного подзапроса или None, если он прошёл."""
        _, rejected = sq.check_subqueries([text], language, "", kept)
        return rejected[0][1] if rejected else None

    def test_word_count(self):
        self.assertIn("слов 1", self.reason("photonics"))
        self.assertIn("слов 5", self.reason("low power analog inference chips"))
        self.assertIsNone(self.reason("analog inference chips"))

    def test_forbidden_characters(self):
        for text in ['edge "ai" chips', "edge (ai) chips", "edge, ai chips", "edge: ai chips",
                     "edge; ai chips", "edge/cloud ai chips"]:
            with self.subTest(text=text):
                self.assertTrue(self.reason(text).startswith("символ"))

    def test_operators(self):
        self.assertEqual(self.reason("edge AND cloud"), "поисковый оператор")

    def test_standalone_year(self):
        self.assertEqual(self.reason("quantum sensing 2025"), "год")
        self.assertIsNone(self.reason("quantum sensing 5000"))

    def test_mature_words_en(self):
        self.assertEqual(self.reason("regulatory sandbox design"), "слово зрелой области regulatory")

    def test_mature_stems_ru(self):
        self.assertEqual(self.reason("стандарты оптических интерфейсов", "ru"),
                         "слово зрелой области стандарты")

    def test_jaccard_duplicate(self):
        self.assertEqual(self.reason("federated learning systems", kept=["federated learning"]),
                         "дубликат по основам: federated learning")
        self.assertIsNone(self.reason("federated unlearning attacks", kept=["federated learning"]))


class GenerateManyTest(IsolatedTest):
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
