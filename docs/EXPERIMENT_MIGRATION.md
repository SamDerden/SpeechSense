# Политика переноса исследований

Ни один исходный эксперимент не удаляется из-за низких метрик, дублирования идеи или реорганизации. Сохраняются код, порядок ячеек, гиперпараметры, seed, выводы, графики, метрики, ошибки запуска и ссылки на веса. Исходники неизменны; адаптированные версии отдельно. Не выдавать восстановленный код за оригинал. Не публиковать персональные данные, токены и материалы без прав публикации.

Источники: WebArchive ru_sentiment_pipeline_adapted_gzip_robust; Трансформеры; Обработка естественного языка. Тест TTS; Google Colab (архив страницы входа, содержимое недоступно). Оригинальные Colab: https://colab.research.google.com/drive/1E_1vjOMYKR2llJmLGfEf3htQW2YUfyBC ; https://colab.research.google.com/drive/1VwnuHub6WizppBC0KRaHjF9hESAYbyJb ; https://colab.research.google.com/drive/1Q6nLxjPtX_qcTDywMlahooNajdFi56gc ; https://colab.research.google.com/drive/1Fj5aD7agSXCyigYYvlcPv_BX7aXdKxoH .

Предварительно подтверждены по отображённому содержимому архивов:
- EDA и предобработка, BoW, binary BoW, TF-IDF, word/char n-grams.
- MultinomialNB, ComplementNB, LogisticRegression, LinearSVC, CalibratedClassifierCV.
- TextCNN, BiLSTM, Navec, FastText, class weights, augmentation, focal loss, lemmatization, cross-entropy; все комбинации и пронумерованные эксперименты подлежат инвентаризации.
- cointegrated/rubert-tiny2, DeepPavlov/rubert-base-cased, blanchefort/rubert-base-cased-sentiment-rusentiment, FacebookAI/xlm-roberta-base, sberbank-ai/ruRoberta-large; augmentation, weighted loss, ensembles, CLS+mean pooling, cosine warmup, different seeds; все остальные варианты подлежат инвентаризации.
- Whisper base/small, TTS.

Для каждого эксперимента: ID, source notebook, cell/section, model, preprocessing, dataset/split, hyperparameters, training settings, metrics, outputs, artifact, status (original / recovered / reproduced / blocked). Отдельно хранить исторические validation-результаты и новые test-результаты. Не считать перенос завершённым до сверки всех ячеек с оригинальными ipynb. WebArchive не гарантирует наличие всех ячеек и выходов.
