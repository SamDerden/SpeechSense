# Реестр исходных экспериментов (первичная инвентаризация заголовков)

Статус: обнаружены в сохранённом HTML, не означает успешное обучение или перенос кода. Нумерацию оригинала сохраняем; пропуски не заполняем догадками.

## Нейросети
- TextCNN baseline
- BiLSTM + Navec + Focal Loss
- 2 TextCNN + Class Weights
- 3 TextCNN + Augmentation Neutral and Negative
- 4 TextCNN + Augmentation + Focal Loss
- 5 TextCNN + Navec
- 6 TextCNN + Navec + Focal Loss
- 7 BiLSTM + Navec (CrossEntropy)
- 9 TextCNN + Lemmatization + Augmentation
- 11 TextCNN + Neutral-only Augmentation
- 12 TextCNN + Focal Loss gamma tuning
- 13 TextCNN + FastText (два одноимённых раздела, сохранить оба)

## Трансформеры
- Пять исходных моделей (см. docs/EXPERIMENT_MIGRATION.md)
- Back-Translation, contextual replacement, class weights, переобучение top-3
- Cosine Annealing with Warmup, ансамбль top-3
- Differential Learning Rates
- Label Smoothing
- Gradient Clipping
- Custom Classifier Head
- CLS + Mean Pooling
- Active Learning
- Curriculum Learning
- Mixup
- Threshold Tuning
- Ensemble with Different Seeds
- Knowledge Distillation
- Multi-Task Learning
- Adversarial Training
- Сравнение ячеек 15–27 и ансамбль vs лучшая техника

## Классика
MultinomialNB (BoW 1–2), ComplementNB (TF-IDF 1–2), LogReg (TF-IDF 1–2), LinearSVC (TF-IDF 1–2), LinearSVC (word+char); дополнительные конфигурации из обоих архивов подлежат сверке по ячейкам.

## ASR/TTS
Whisper base/small, TTS. Четвёртый WebArchive содержит страницу входа; требуется оригинальный ipynb.
