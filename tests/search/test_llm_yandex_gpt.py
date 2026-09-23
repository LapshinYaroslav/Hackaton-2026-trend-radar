"""Тесты обёртки над YandexGPT: HTTP подменяется, реальных запросов нет."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests

from search import llm_yandex_gpt as llm

ENV = {"YANDEX_API_KEY": "секретный-ключ", "YANDEX_FOLDER_ID": "b1gtest",
       "YANDEX_GPT_MODEL": "yandexgpt-lite"}
BODY = {"alternatives": [{"message": {"role": "assistant", "text": "ответ модели"},
                          "status": "ALTERNATIVE_STATUS_FINAL"}],
        "usage": {"inputTextTokens": "10", "completionTokens": "5", "totalTokens": "15"},
        "modelVersion": "23.10.2024"}


class FakeResponse:
    """Подделка requests.Response: только то, что использует ask_llm."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self) -> dict:
        return self._payload


class AskLlmTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_file = Path(tmp.name) / "llm_calls.jsonl"

    def _call(self, **post_kwargs):
        """Вызывает ask_llm с подменёнными окружением, логом и requests.post."""
        with patch.dict("os.environ", ENV), \
             patch.object(llm, "LOG_FILE", self.log_file), \
             patch.object(llm.SESSION, "post", **post_kwargs) as post:
            result = llm.ask_llm("системный промпт", "пользовательский промпт", purpose="subqueries")
        return result, post

    def test_request_body(self):
        """json_object на верхнем уровне, temperature внутри completionOptions, ключ в заголовке."""
        _, post = self._call(return_value=FakeResponse({"result": BODY}))
        body = post.call_args.kwargs["json"]
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(post.call_args.args[0], llm.COMPLETION_URL)
        self.assertIs(body["json_object"], True)
        self.assertNotIn("json_object", body["completionOptions"])
        self.assertEqual(body["completionOptions"]["temperature"], 0.3)
        self.assertEqual(body["completionOptions"]["maxTokens"], "2000")
        self.assertIs(body["completionOptions"]["stream"], False)
        self.assertEqual(body["modelUri"], "gpt://b1gtest/yandexgpt-lite/latest")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(headers["Authorization"], "Api-Key секретный-ключ")
        self.assertEqual(headers["x-folder-id"], "b1gtest")

    def test_reads_answer(self):
        """Из ответа берутся текст, версия модели и токены."""
        result, _ = self._call(return_value=FakeResponse({"result": BODY}))
        self.assertEqual(result["text"], "ответ модели")
        self.assertEqual(result["model_version"], "23.10.2024")
        self.assertEqual(result["usage"]["totalTokens"], "15")
        self.assertIsNone(result["error"])

    def test_answer_without_result_wrapper(self):
        """Тело без обёртки result читается так же."""
        result, _ = self._call(return_value=FakeResponse(BODY))
        self.assertEqual(result["text"], "ответ модели")

    def test_network_error(self):
        """Ошибка сети возвращается в error, а не исключением."""
        result, _ = self._call(side_effect=requests.ConnectionError("нет сети"))
        self.assertIsNone(result["text"])
        self.assertIn("ConnectionError", result["error"])

    def test_broken_answer_shape(self):
        """Ответ без alternatives не роняет вызов."""
        result, _ = self._call(return_value=FakeResponse({"result": {}}))
        self.assertIsNone(result["text"])
        self.assertIn("KeyError", result["error"])

    def test_log_without_api_key(self):
        """В лог пишется вызов, но не ключ; кириллица не экранируется."""
        self._call(return_value=FakeResponse({"result": BODY}))
        raw = self.log_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw), 1)
        self.assertNotIn("секретный-ключ", raw[0])
        self.assertIn("системный промпт", raw[0])
        record = json.loads(raw[0])
        self.assertEqual(record["purpose"], "subqueries")
        self.assertEqual(record["model_version"], "23.10.2024")
        self.assertEqual(record["response_text"], "ответ модели")
        self.assertIsNone(record["error"])

    def test_retry_on_429(self):
        """429 -> пауза и повтор, со второй попытки ответ принимается."""
        with patch.object(llm.time, "sleep") as sleep:
            result, post = self._call(side_effect=[FakeResponse({}, status_code=429),
                                                   FakeResponse({"result": BODY})])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(sleep.call_args.args[0], llm.RETRY_DELAY)
        self.assertEqual(result["text"], "ответ модели")
        self.assertIsNone(result["error"])

    def test_429_after_all_attempts(self):
        """429 на всех попытках: ошибка возвращается в поле error."""
        with patch.object(llm.time, "sleep"):
            result, post = self._call(side_effect=[FakeResponse({}, status_code=429)] * llm.RETRY_ATTEMPTS)
        self.assertEqual(post.call_count, llm.RETRY_ATTEMPTS)
        self.assertIn("429", result["error"])

    def test_concurrency_limit(self):
        """Семафор ограничивает число одновременных обращений к API."""
        self.assertEqual(llm.CALL_LIMIT._value, llm.MAX_CONCURRENT_CALLS)
        self.assertLessEqual(llm.MAX_CONCURRENT_CALLS, 10)

    def test_model_not_allowed(self):
        """Модель вне белого списка — ошибка до обращения к API."""
        with patch.dict("os.environ", {**ENV, "YANDEX_GPT_MODEL": "gpt-4"}):
            with self.assertRaises(ValueError):
                llm.build_model_uri()

    def test_timeout_retried_once_with_same_body(self):
        """Таймаут: один повтор с тем же телом запроса, затем ответ."""
        result, post = self._call(side_effect=[requests.ReadTimeout("медленно"), FakeResponse(BODY)])
        self.assertIsNone(result["error"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"], post.call_args_list[1].kwargs["json"])

    def test_second_timeout_is_error(self):
        """Два таймаута подряд: ошибка в ответе, третьей попытки нет."""
        result, post = self._call(side_effect=[requests.ReadTimeout("раз"), requests.ReadTimeout("два")])
        self.assertIn("ReadTimeout", result["error"])
        self.assertIsNone(result["model_version"])
        self.assertEqual(post.call_count, 2)

    def test_explicit_model_uri_without_branch(self):
        """Явное имя модели идёт в URI без /latest, старый псевдоним — с ним."""
        with patch.dict("os.environ", {**ENV, "YANDEX_GPT_MODEL": "yandexgpt-5-pro"}):
            self.assertEqual(llm.build_model_uri(), "gpt://b1gtest/yandexgpt-5-pro")
        with patch.dict("os.environ", {**ENV, "YANDEX_GPT_MODEL": "yandexgpt"}):
            self.assertEqual(llm.build_model_uri(), "gpt://b1gtest/yandexgpt/latest")

    def test_missing_api_key(self):
        """Без ключа в .env вызов не делается."""
        with patch.dict("os.environ", {**ENV, "YANDEX_API_KEY": ""}):
            with self.assertRaises(ValueError):
                llm.ask_llm("с", "п", purpose="subqueries")


if __name__ == "__main__":
    unittest.main()
