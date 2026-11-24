# Контракт API SpeechSense

Маппинг обученных моделей проекта: 0 neutral, 1 positive, 2 negative.
Для внешней модели требуется явное проверенное соответствие её config.id2label; числовые метки нельзя угадывать.

Успешный ответ (пример, не результат измерения):
```json
{"status":"ok","text":"Спасибо, всё понравилось","sentiment_id":1,"sentiment":"positive","confidence":0.9,"probabilities":{"neutral":0.07,"positive":0.9,"negative":0.03}}
```

Статусы ood, asr_error, invalid_input, model_error не являются классами тональности. Для них sentiment_id, sentiment, confidence и probabilities равны null.
Проверять три вероятности, конечность, диапазон и сумму, соответствие argmax метке. Не подменять обученную модель публичной без явного режима демо.

## Дефекты оригинала Bot+Server.ipynb
- Ячейка 3: LABEL_0/1/2 соответствуют negative/neutral/positive вместо согласованного порядка; обрезка символов вместо токенов.
- Ячейки 4–5: повторное определение API, нет OOD и распределения вероятностей.
- Ячейка 6: file_id2 вместо file_id, неверный отступ finally, жёсткий localhost.
- Исправления должны войти в новые server/ и telegram_bot/, оригинал сохраняется.
