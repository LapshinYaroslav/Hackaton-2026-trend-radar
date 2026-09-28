# Склейка дублей в ТОП-15: проверка на слепой разметке и решение команды

Задача К, 26–27.09.2026. Код — `pipeline/dedup.py`, вызов — `pipeline/run_query.py` (этап «Склейка дублей»).

## Зачем

В готовых примерах одна технология занимала несколько мест ТОП-15 под разными названиями: в «технологиях в ИИ»
климатическое моделирование с ИИ — места 2, 3, 4 и 15; в «кибербезопасности» обнаружение угроз с ИИ — места 1 и 11.

## Метод

**Ключ термина** (`term_en`): нижний регистр; `-`, `/`, `_` → пробел; прочая пунктуация убирается; убираются фраза
`artificial intelligence` и стоп-слова (`ai, driven, powered, enhanced, based, assisted, enabled, using, for, of, the,
a, an, with, and, in, on, to, system(s), approach(es), method(s), framework(s), tool(s)`); стемминг Snowball (english);
ключ — множество основ.

**Проход:** по оценённым кандидатам в порядке убывания score; кандидат, дублирующий уже принятого, не занимает место,
а попадает в `variants` принятого (`[{term_en, score}]`); в исключённых он получает причину `duplicate_of`; ТОП-15
добирается следующими по score, правило порога не меняется.

**Проверка:** слепые разметки пяти экспериментов (Н — 64, Н2 — 116, Г — 206, Д — 163, Ж — 98 терминов) и ключи
слепоты. Все пары терминов внутри одной темы одного файла, которые вариант признаёт дублями:
- верная склейка — одинаковый `canonical_concept`;
- ошибочная — разный `canonical_concept`;
- непроверяемая — у обоих метка noise, mature или umbrella и «корзинный» `canonical_concept`
  (`title phrase…`, `product pitch…`, `off-topic…`, `general ML methods`, `mature AI tech`, `AI umbrella`).

**Правило, заданное до запуска:** 1) ошибочных склеек, где хотя бы один термин `plausible_signal` или `uncertain`, —
не больше 1; 2) ошибочных среди проверяемых — не больше 10 %; 3) на примерах: климатические термины «ai» — одна строка,
места 1 и 11 «cybersec» — одна строка. Порог и списки после запуска не подбирались.

## Три варианта решения «дубль»

- **К1:** ключи равны; или меньший ключ из ≥ 2 основ целиком входит в больший; или Жаккар ≥ 0.6.
- **(а):** только полное равенство ключей из ≥ 1 основы.
- **(б):** пары с хотя бы одной общей основой ключа — решение YandexGPT (`yandexgpt-5-pro`, температура 0, кэш по паре),
  промпт дословно: «Термин 1: «{a}». Термин 2: «{b}». Это одна и та же технология, названная по-разному (синонимы,
  другой порядок слов, общие слова вроде AI-driven, system, based)? Ответ «нет», если один термин — частный случай,
  разновидность, применение или более широкий класс другого, или если это разные технологии. Ответь одним словом:
  да или нет.» Ответ не «да» и не «нет» — «нет». Кандидатных пар по разметке — 1297, вызовов — 1462 (с примерами).

| вариант | верные | ошибочные | непроверяемые | ошибочных с plausible/uncertain (≤ 1) | ошибочных среди проверяемых (≤ 10 %) | климат «ai» — одна строка | «cybersec» 1 и 11 — одна строка |
|---|---|---|---|---|---|---|---|
| К1 | 49 | 41 | 2 | 6 | 41 из 90 = 45.6 % | да | да |
| (а) | 10 | 0 | 0 | 0 | 0 из 10 = 0 % | нет (2 строки) | нет |
| (б) | 43 | 8 | 0 | 4 | 8 из 51 = 15.7 % | нет (2 строки) | да |

По файлам, верные / ошибочные: К1 — Н 0/0, Н2 5/6, Г 31/23, Д 13/11, Ж 0/1; (а) — Н2 1/0, Г 6/0, Д 3/0;
(б) — Н 2/2, Н2 3/1, Г 26/1, Д 12/0, Ж 0/4.

### Все ошибочные склейки варианта (б)

| файл | термин 1 | термин 2 | метки | canonical_concept |
|---|---|---|---|---|
| Н | ai-powered real-time threat detection | adaptive threat defense | noise / noise | AI threat detection / adaptive threat defense |
| Н | Agentic AI | LLM agent | umbrella / umbrella | agentic AI / LLM agents |
| Н2 | deep learning | deep representation learning | umbrella / mature | deep learning / representation learning |
| Г | fusion power | controlled nuclear fusion | plausible_signal / umbrella | commercial fusion power / controlled nuclear fusion |
| Ж | decentralized swarm control | decentralized multi-robot collaboration | mature / noise | swarm control / title phrase robotics |
| Ж | ai-native cybersecurity | defensive cybersecurity ai | umbrella / plausible_signal | AI cybersecurity / AI agent security |
| Ж | defensive cybersecurity ai | ai cybersecurity platform | plausible_signal / uncertain | AI agent security / security-specialized frontier models |
| Ж | defensive cybersecurity ai | ai cybersecurity model | plausible_signal / uncertain | AI agent security / security-specialized frontier models |

## Решение

Правило, заданное до проверки (≤ 1 ошибки с хорошими терминами, ≤ 10%), не выполнено: 16%. Склейка включена решением
команды: повторы одной технологии в ТОП-15 хуже для читателя, чем редкое поглощение узкого термина. Поглощённые
термины не теряются — они показываются под принятым как „также: …“.

## Что включено в продукт

Вариант (б) с одним уточнением из (а): пара с равными непустыми ключами — дубль без вызова YandexGPT; остальные пары
с общей основой ключа решает YandexGPT по промпту выше. Ошибка или таймаут YandexGPT — пара «не дубль», одно
предупреждение в выдаче, прогон не падает. По умолчанию включено (`run_query(dedup=True)`), для сравнения —
`python -m pipeline … --no-dedup`. Скрипт проверки и разметка в репозиторий не входят: разметка — рабочие файлы
команды, без неё проверку не повторить.

## К6: подбор промпта точнее, чем (б)

Задача К6, 27.09.2026. Разметка разделена **до запуска**: подбор — Н, Н2, Г; проверка — Д, Ж. Все варианты посчитаны
одним запуском, промпты после запуска не менялись; примеры в промптах придуманы, не из разметки. Во всех вариантах
пара с равными непустыми ключами — дубль без вызова; остальные пары с общей основой — вызов `yandexgpt-5-pro`,
температура 0, кэш по (вариант, пара). Кандидатных пар — 1460 (разметка и примеры «ai», «cybersec»), вызовов — 3569.

- **B0** — промпт (б) выше (в продукте).
- **B1** — примеры «да/нет»: «Термин 1: «{a}». Термин 2: «{b}». Это одна и та же технология, названная по-разному?
  Правила: «да» — только если термины взаимозаменяемы в аналитическом отчёте (синонимы, аббревиатура, другой порядок
  слов, общие слова вроде AI-driven, based, system). «нет» — если один термин уже, шире, частный случай, применение,
  компонент другого, или это соседние, но разные технологии. Примеры: «retrieval-augmented generation» / «RAG system» —
  да; «LLM-based code generation» / «code generation with large language models» — да; «federated learning» /
  «federated reinforcement learning» — нет; «energy storage» / «battery energy storage» — нет; «AI security» /
  «security of AI agents» — нет; «drones» / «drone delivery» — нет. Ответь одним словом: да или нет.»
- **B2** — тип отношения, JSON: «Термин 1: «{a}». Термин 2: «{b}». Определи отношение между технологиями. Верни JSON
  {"relation": одно из "same" (одна технология, разные формулировки), "narrower" (термин 1 — частный случай,
  разновидность, применение или компонент термина 2), "broader" (термин 1 шире термина 2), "related" (соседние, но
  разные технологии), "different" (не связаны)}. Если сомневаешься между "same" и другим отношением — выбирай другое.»
  Дубль — только "same"; неразборчивый ответ — не дубль.
- **B3** — B2 в обе стороны: дубль, только если оба ответа "same".

| вариант | часть | верные | ошибочные | из них с plausible/uncertain | доля ошибочных | климат «ai», строк | «cybersec» 1 и 11 вместе |
|---|---|---|---|---|---|---|---|
| B0 | подбор | 32 | 4 | 1 | 11.1 % | | |
| B0 | проверка | 14 | 4 | 3 | 22.2 % | 2 | да |
| B1 | подбор | 24 | 1 | 1 | 4.0 % | | |
| B1 | проверка | 13 | 3 | 2 | 18.8 % | 2 | нет |
| B2 | подбор | 14 | 1 | 1 | 6.7 % | | |
| B2 | проверка | 8 | 3 | 2 | 27.3 % | 2 | нет |
| B3 | подбор | 13 | 1 | 1 | 7.1 % | | |
| B3 | проверка | 8 | 0 | 0 | 0 % | 2 | нет |

**Правило выбора (задано до запуска), на проверке:** ошибочных с plausible/uncertain меньше, чем у B0 (3); доля
ошибочных ≤ 10 %; верных не меньше 60 % от B0 (≥ 8.4).
- B1: 2 < 3 — да; 18.8 % — нет.
- B2: 2 < 3 — да; 27.3 % — нет.
- B3: 0 < 3 — да; 0 % — да; 8 < 8.4 — нет.

**Итог:** кандидатов нет — в продукте остаётся B0.

### Все ошибочные склейки B0 (в продукте)

| часть | файл | термин 1 | термин 2 | метки | canonical_concept |
|---|---|---|---|---|---|
| подбор | Г | fusion power | controlled nuclear fusion | plausible_signal / umbrella | commercial fusion power / controlled nuclear fusion |
| подбор | Н | ai-powered real-time threat detection | adaptive threat defense | noise / noise | AI threat detection / adaptive threat defense |
| подбор | Н | Agentic AI | LLM agent | umbrella / umbrella | agentic AI / LLM agents |
| подбор | Н2 | deep learning | deep representation learning | umbrella / mature | deep learning / representation learning |
| проверка | Ж | decentralized swarm control | decentralized multi-robot collaboration | mature / noise | swarm control / title phrase robotics |
| проверка | Ж | ai-native cybersecurity | defensive cybersecurity ai | umbrella / plausible_signal | AI cybersecurity / AI agent security |
| проверка | Ж | defensive cybersecurity ai | ai cybersecurity platform | plausible_signal / uncertain | AI agent security / security-specialized frontier models |
| проверка | Ж | defensive cybersecurity ai | ai cybersecurity model | plausible_signal / uncertain | AI agent security / security-specialized frontier models |
