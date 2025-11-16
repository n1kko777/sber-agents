# Отчёт по заданию «Мониторинг QA»

- Проект: RAG-ассистент Сбербанка — Telegram-бот с Retrieval-Augmented Generation для ответов на вопросы по документам Сбербанка о кредитах и вкладах (README.md:1-3)
- Вариант: базовый
- Модели и провайдеры:
  - RAG: LLM `openai/gpt-oss-20b:free` (OpenRouter), Query Transform `openai/gpt-oss-20b:free`, Embedding `text-embedding-3-large`
  - RAGAS evaluation: LLM `openai/gpt-oss-20b:free`, Embedding `text-embedding-3-large`

## Датасет

- Способ создания: автоматический синтез из документов (LLM генерирует вопросы/ответы по чанкам) + загрузка готовых Q&A из JSON; всё собирается в один файл и может быть загружено в LangSmith (README.md:300-317)
- Размер: 6 примеров (datasets/06-rag-qa-dataset.json)
- Скриншот LangSmith датасета: [screenshots/базовый вариант с openrouter.png](screenshots/базовый вариант с openrouter.png)
- Примеры Q&A:
  - Вопрос: «Кто является сторонами по Договору в условиях предоставления потребительского кредита?» Ответ: «Сторонами по Договору являются Кредитор и Заемщик или Созаемщики.» (datasets/06-rag-qa-dataset.json:3-12)
  - Вопрос: «Что происходит с поступившими средствами, если они совпадают с Платежной датой и нет просроченной задолженности?» Ответ: «Поступившие на счет Кредитора средства направляются на погашение Срочной задолженности по Кредиту и уплату срочных процентов за пользование Кредитом.» (datasets/06-rag-qa-dataset.json:15-24)

## Оценка через RAGAS

- Метрики: Faithfulness, Answer Relevancy, Answer Correctness, Answer Similarity, Context Recall, Context Precision (README.md:351-358)

## Выводы о качестве

- В демонстрационном прогоне RAGAS (README.md:333-348) высокие Faithfulness/Correctness/Similarity (≥0.82) показывают, что ответы опираются на retrieved контекст и совпадают с эталоном.
- Более низкие Answer Relevancy и Context Recall (0.65 и 0.75) указывают, что узким местом остаётся поиск/отбор контекста; дальнейшие улучшения стоит фокусировать на ретривере и промпте query transform.
