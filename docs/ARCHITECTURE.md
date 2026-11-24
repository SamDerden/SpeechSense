# Архитектура SpeechSense

Telegram (aiogram) → HTTP API (FastAPI) → Whisper ASR (для аудио) → предобработка → OOD → классификатор тональности → JSON → Telegram. Бот и сервер разделены. OOD не является нейтральным классом. Низкая уверенность и ошибка ASR обрабатываются отдельно.

## Структура
- data/ — исходные данные вне Git, схема, контрольные суммы и инструкция загрузки.
- notebooks/01_EDA.ipynb
- notebooks/02_classical_models.ipynb
- notebooks/03_neural_networks.ipynb
- notebooks/04_transformers.ipynb
- notebooks/05_model_optimization.ipynb — только межсемейные ансамбли и сравнение; Optuna LogReg и fastText supervised в 02, все эксперименты с трансформерами в 04, FastText embeddings в 03.
- notebooks/06_ASR_TTS.ipynb
- notebooks/07_OOD_detection.ipynb
- notebooks/08_final_comparison.ipynb
- notebooks/09_final_homework.ipynb — объединённый воспроизводимый блокнот
- notebooks/originals/ — неизменённые исходные исследования, если доступны; архивы вне Git при превышении лимитов.
- src/ — dataset, preprocessing, metrics, sentiment, asr, ood, pipeline.
- models/ — manifest, конфигурации, label mapping, ссылки и контрольные суммы весов.
- reports/ — EDA, результаты по семействам, анализ ошибок, итоговое сравнение, реестр экспериментов.
- server/ — FastAPI, схемы и endpoints.
- telegram_bot/ — aiogram, handlers, config.
- tests/ — модульные и сквозные тесты.
- docs/ — архитектура, требования, политика переноса.
- requirements.txt, .env.example, .gitignore, Dockerfile.

## Протокол
Единый train/validation/test; выбор модели и гиперпараметров только на validation; test для итоговой оценки. Macro-F1 основная метрика, accuracy дополнительная; weighted-F1, per-class precision/recall/F1, PR-AUC OvR, confusion matrix, скорость и размер. Исторические метрики отдельно. Маппинг 0 neutral, 1 positive, 2 negative. 
