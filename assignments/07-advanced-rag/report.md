# Отчёт по экспериментам Advanced RAG

## 1. Методика и данные
- Перед каждым прогоном заново индексировались 377 PDF-чанков и 212 Q&A из JSON (всего 589 элементов), чтобы гарантировать воспроизводимость результатов (logs/bot.log:1873, logs/bot.log:1877, logs/bot.log:1879).
- Во всех тестах использовались HuggingFace-эмбеддинги `intfloat/multilingual-e5-base` на устройстве MPS и LangSmith-трейсинг (logs/bot.log:1864, logs/bot.log:1866).
- Запросы переписывались вспомогательной LLM `openai/gpt-oss-120b`, а ответы генерировал `openai/gpt-oss-20b:free` (logs/bot.log:1906, logs/bot.log:1908, logs/bot.log:2092, logs/bot.log:2094, logs/bot.log:2629, logs/bot.log:2635).
- Для оценки запускался скрипт `evaluate_with_ragas` (3 шага: генерация ответов, RAGAS, загрузка в LangSmith) и фиксировались метрики `faithfulness`, `answer_relevancy`, `answer_correctness`, `answer_similarity`, `context_recall`, `context_precision` при использовании `openai/gpt-oss-120b` + тех же эмбеддингов (logs/bot.log:1900, logs/bot.log:2086, logs/bot.log:2622).
- Каждый эксперимент использовал одинаковый батч из 6 вопросов из датасета `07-advanced-rag` (logs/bot.log:1920, logs/bot.log:2108, logs/bot.log:2652).

## 2. Эксперимент 1 — Semantic Retrieval (базовый)
### Конфигурация
- Режим поиска: чистый семантический FAISS-ретривер (logs/bot.log:1863, logs/bot.log:1884).
- Эмбеддинги: HuggingFace `intfloat/multilingual-e5-base` (logs/bot.log:1864).
- LLM-цепочка: query transform через `openai/gpt-oss-120b`, генерация ответов через `openai/gpt-oss-20b:free` (logs/bot.log:1906, logs/bot.log:1908).

### Метрики RAGAS
| metric | value |
| --- | --- |
| faithfulness | 0.633 |
| answer_relevancy | 0.745 |
| answer_correctness | 0.630 |
| answer_similarity | 0.921 |
| context_recall | 1.000 |
| context_precision | 0.738 |

Источник: logs/bot.log:2026, logs/bot.log:2027, logs/bot.log:2028, logs/bot.log:2029, logs/bot.log:2030, logs/bot.log:2031.

### Наблюдения
Семантический поиск даёт высокую полноту (recall 1.0) и достойную точность контекста (0.738), однако качество ответов ограничено: faithfulness 0.633 и correctness 0.630 сигнализируют о частичной галлюцинации модели при неоднозначных вопросах.

## 3. Эксперимент 2 — Hybrid Retrieval (Semantic + BM25)
### Конфигурация
- Режим: ансамбль из векторного поиска и BM25 с равными весами (logs/bot.log:2047, logs/bot.log:2051, logs/bot.log:2072).
- Эмбеддинги и LLM как в первом эксперименте (logs/bot.log:2050, logs/bot.log:2092, logs/bot.log:2094).
- Гиперпараметры: semantic_k = 10, bm25_k = 10 (logs/bot.log:2051).

### Метрики RAGAS
| metric | value |
| --- | --- |
| faithfulness | 0.551 |
| answer_relevancy | 0.759 |
| answer_correctness | 0.610 |
| answer_similarity | 0.917 |
| context_recall | 1.000 |
| context_precision | 0.703 |

Источник: logs/bot.log:2246, logs/bot.log:2247, logs/bot.log:2248, logs/bot.log:2249, logs/bot.log:2250, logs/bot.log:2251.

### Наблюдения
Добавление BM25 повысило релевантность ответов (0.759 против 0.745) за счёт попадания в редкие формулировки, но faithfulness просел до 0.551, а точность контекста — до 0.703. Ансамбль часто вытягивал более «шумные» отрывки, и LLM отвечал менее уверенно, несмотря на неизменное покрытие (recall 1.0).

## 4. Эксперимент 3 — Hybrid + Cross-encoder Reranker
### Конфигурация
- Базовая часть как в Hybrid (semantic_k = 10, bm25_k = 10, веса 0.5/0.5) (logs/bot.log:2586, logs/bot.log:2587, logs/bot.log:2608, logs/bot.log:2609).
- После объединённой выдачи применялся кросс-энкодер `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` с top-k=3 (logs/bot.log:2588, logs/bot.log:2589, logs/bot.log:2631, logs/bot.log:2634).
- Остальной стек (эмбеддинги и LLM) идентичен предыдущим экспериментам (logs/bot.log:2583, logs/bot.log:2629, logs/bot.log:2635).

### Метрики RAGAS
| metric | value |
| --- | --- |
| faithfulness | 0.850 |
| answer_relevancy | 0.748 |
| answer_correctness | 0.765 |
| answer_similarity | 0.920 |
| context_recall | 1.000 |
| context_precision | 1.000 |

Источник: logs/bot.log:2717, logs/bot.log:2718, logs/bot.log:2719, logs/bot.log:2720, logs/bot.log:2721, logs/bot.log:2722.

### Наблюдения
Кросс-энкодер отфильтровал шумные документы и оставил только три наиболее качественных абзаца, что привело к резкому росту faithfulness до 0.85 и correctness до 0.765 при одновременном достижении идеальной точности по контексту (context_precision = 1.0). Незначительное падение answer_relevancy связано с более узким контекстным окном, но качество ответа стало существенно выше.

## 5. Сравнительный анализ
| metric | semantic | hybrid | hybrid+reranker |
| --- | --- | --- | --- |
| faithfulness | 0.633 | 0.551 | **0.850** |
| answer_relevancy | 0.745 | **0.759** | 0.748 |
| answer_correctness | 0.630 | 0.610 | **0.765** |
| answer_similarity | 0.921 | 0.917 | **0.920** |
| context_recall | 1.000 | 1.000 | 1.000 |
| context_precision | 0.738 | 0.703 | **1.000** |

- **Релевантность**: гибридный режим показывает самый высокий показатель relevancy, однако выигрывает за счёт большего количества возвращаемых фрагментов, что вредит точности.
- **Фактическая точность**: reranker обеспечивает +0.22 к faithfulness относительно baseline и +0.14 к answer_correctness, гарантируя отсутствие нерелевантных источников (context_precision = 1.0).
- **Стабильность**: все режимы удерживают context_recall = 1.0, но только reranker способен одновременно поддерживать высокую полноту и строгую фильтрацию контекстов.

## 6. Выводы и рекомендации
1. **Лучшей конфигурацией стал Hybrid + Reranker**, поскольку он единственный обеспечил одновременный рост faithfulness (0.850) и answer_correctness (0.765) при идеальном качестве контекстов (logs/bot.log:2717, logs/bot.log:2719, logs/bot.log:2722).
2. **Гибрид без reranker** полезен лишь в сценариях, где требуется максимум релевантных фрагментов для ручного анализа. Для автоматических ответов он увеличивает шум и снижает доверие (logs/bot.log:2246, logs/bot.log:2251).
3. **Дальнейшие шаги**:
   - протестировать более лёгкие кросс-энкодеры (например, bge-reranker-base) для уменьшения латентности без потери faithfulness;
   - повысить устойчивость answer_relevancy в конфигурации с reranker путём лёгкого увеличения reranker top-k (например, 5) и добавления переформулировок запросов для редких формулировок;
   - расширить датасет вопросов, чтобы уменьшить дисбаланс между релевантностью и точностью (6 примеров недостаточно для статистической значимости).
