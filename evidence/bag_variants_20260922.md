# Четыре версии рыночной доли

Признаки: `axis_features_20260921.csv`, 160 строк, 100 сигналов, 60 мейнстримов. Документы: пересбор TechCrunch с полной выдачей, 19705 документов в шести окнах периода. Сети в этом прогоне нет: счётчики посчитаны заранее из сырых ответов.

Версии сопоставления названия с документом:

- **текущая** — фильтра нет вовсе, счётчик X-WP-Total: документ засчитан, если его вернул поиск TechCrunch;
- **точная фраза** — название встречается целиком подряд;
- **все слова где угодно** — все значимые основы названия есть в документе в любом порядке и на любом расстоянии;
- **все слова в одном абзаце** — то же, но внутри одного абзаца; заголовок считается отдельным абзацем.

Стоп-слова выброшены списком sklearn, слова приведены стеммером Портера (`experiments/explore/porter.py`, своя реализация: библиотеки со стеммером в окружении нет; проверена на словаре примеров из статьи 1980 года). Регистр, дефисы, HTML-сущности и притяжательное «'s» нормализованы.

Усечённые строки (2: больше 300 документов в одном окне) из всех трёх строгих версий исключены — по ним известен только нижний предел.

## 1. Доля попаданий по классам

«Доля попаданий» — сколько документов выдачи вариант засчитывает. Первая строка показывает, что текущая версия не фильтрует ничего.

| вариант | признак | доля попаданий, сигналы | медиана по технологии, сигналы | доля попаданий, мейнстримы | медиана по технологии, мейнстримы | доля попаданий, все | медиана по технологии, все |
|---|---|---|---|---|---|---|---|
| точная фраза | share_news_wordmatch_strict | 0.039 | 0.0 | 0.117 | 0.0 | 0.065 | 0.0 |
| все слова где угодно | share_news_wordmatch_bag | 0.88 | 1.0 | 0.732 | 1.0 | 0.83 | 1.0 |
| все слова в одном абзаце | share_news_wordmatch_bag2 | 0.123 | 0.082 | 0.236 | 0.211 | 0.161 | 0.109 |

## 2. Четыре версии рядом

«AUC признака» — сама доля как одиночный признак. «AUC набора» — S2, где share_news_wordmatch заменена на эту версию; разность парная, внутри повтора.

### 2.1. В составе S2

| версия доли | признак | заполнено | AUC признака | AUC набора | разность с базой (парно) | повторов лучше базы |
|---|---|---|---|---|---|---|
| текущая: все документы выдачи | share_news_wordmatch | 160 | 0.671 | 0.794 ± 0.011 | — | — |
| точная фраза | share_news_wordmatch_strict | 155 | 0.416 | 0.645 ± 0.024 | -0.1487 ± 0.0196 | 0/10 |
| все слова где угодно | share_news_wordmatch_bag | 158 | 0.67 | 0.783 ± 0.013 | -0.0115 ± 0.0050 | 0/10 |
| все слова в одном абзаце | share_news_wordmatch_bag2 | 157 | 0.576 | 0.719 ± 0.017 | -0.0749 ± 0.0138 | 0/10 |

### 2.2. В составе S2 + age_first_arxiv

| версия доли | признак | заполнено | AUC признака | AUC набора | разность с базой (парно) | повторов лучше базы |
|---|---|---|---|---|---|---|
| текущая: все документы выдачи | share_news_wordmatch | 160 | 0.671 | 0.805 ± 0.010 | — | — |
| точная фраза | share_news_wordmatch_strict | 155 | 0.416 | 0.686 ± 0.020 | -0.1185 ± 0.0171 | 0/10 |
| все слова где угодно | share_news_wordmatch_bag | 158 | 0.67 | 0.795 ± 0.010 | -0.0092 ± 0.0043 | 0/10 |
| все слова в одном абзаце | share_news_wordmatch_bag2 | 157 | 0.576 | 0.739 ± 0.017 | -0.0656 ± 0.0139 | 0/10 |

## 3. Связь версий между собой

| признак | с чем | ро, сигналы | n, сигналы | ро, мейнстримы | n, мейнстримы |
|---|---|---|---|---|---|
| share_news_wordmatch | share_news_wordmatch_strict | -0.15 | 96 | -0.069 | 59 |
| share_news_wordmatch | share_news_wordmatch_bag | 0.999 | 99 | 0.999 | 59 |
| share_news_wordmatch | share_news_wordmatch_bag2 | 0.72 | 98 | 0.639 | 59 |
| share_news_wordmatch_bag | share_news_wordmatch_bag2 | 0.724 | 98 | 0.646 | 59 |

AUC всех четырёх и счётчиков под ними:

| признак | строки | сигналов | мейнстримов | AUC | отклонение от 0.5 | z |
|---|---|---|---|---|---|---|
| share_news_wordmatch | все строки | 100 | 60 | 0.671 | 0.171 | 3.618 |
| share_news_wordmatch_strict | все строки | 96 | 59 | 0.416 | 0.084 | 1.758 |
| share_news_wordmatch_bag | все строки | 99 | 59 | 0.67 | 0.17 | 3.577 |
| share_news_wordmatch_bag2 | все строки | 98 | 59 | 0.576 | 0.076 | 1.598 |
| n_market | все строки | 100 | 60 | 0.538 | 0.038 | 0.802 |
| n_market_strict | все строки | 99 | 59 | 0.406 | 0.094 | 1.964 |
| n_market_bag | все строки | 99 | 59 | 0.542 | 0.042 | 0.888 |
| n_market_bag2 | все строки | 99 | 59 | 0.466 | 0.034 | 0.719 |

## 4. Пятнадцать документов: засчитаны в bag, но не точной фразой

Выборка случайная из всех таких документов окон периода, random_state=42. «Слова названия» — значимые основы, которые искались.

| tech_key | класс | слова названия | заголовок | в одном абзаце |
|---|---|---|---|---|
| ai access control system | сигнал | access ai control | Controversial drone company Xtend leans into defense with new $40 million round |  |
| ai access control system | сигнал | access ai control | This Week in Apps: AI-powered productivity apps, US weighs TikTok ban, SVB crash boosts crypto apps |  |
| ai infrastructure security | сигнал | ai infrastructur secur | Two arrested for smuggling AI chips to China — Nvidia says no to kill switches |  |
| ai process control | сигнал | ai control process | This Week in AI: AI gets creative in the kitchen |  |
| ai process control | сигнал | ai control process | Ford lifts the hood on its EV business, Turo updates its IPO filing and Waymo releases a safety case for AVs |  |
| ai process control | сигнал | ai control process | 9 ways founders can bring automation to healthcare |  |
| ai process control | сигнал | ai control process | UK consults on opt-out model for training AIs on copyrighted content |  |
| banking as a service | мейнстрим | bank servic | Predatory loan apps in India rake in huge fees, and are driving some users to suicide | да |
| conditional payment system | сигнал | condit payment | Here are all the companies from Day 2 of Y Combinator’s Summer 2021 Demo Day |  |
| edge cloud computing orchestration | сигнал | cloud comput edg orchestr | The rise of cybersecurity debt |  |
| industrial ai agent | сигнал | agent ai industri | This Bay Area startup is using AI to help families navigate long-term care planning |  |
| industrial time series foundation model | сигнал | foundat industri model seri time | ​Why a downturn can separate the recession-proof startups​ from the ‘hacks’ |  |
| reception robot | мейнстрим | recept robot | In 2021, space investors watched stars form in real time |  |
| robotic process automation | мейнстрим | autom process robot | Is the US labor shortage the big break AI needs? |  |
| robotic process automation | мейнстрим | autom process robot | Tiger’s bullish robotic investments | да |
