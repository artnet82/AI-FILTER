from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import threading
from queue import Queue

app = Flask(__name__)

# Загрузка предварительно обученной модели BERT
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Загрузка данных с запрещенными словами
forbidden_words = ['запрещенное', 'слово', 'недопустимый']

# Очередь событий для хранения запросов
event_queue = Queue()

def process_content(content):
    # Предварительная обработка текста
    input_ids = tokenizer.encode(content, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])

    # Классификация текста с помощью BERT
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.argmax(outputs.logits, dim=1).item()

    # Проверка наличия запрещенных слов
    if predictions == 1:
        for word in forbidden_words:
            if word in content:
                return f"Обнаружено запрещенное слово: {word}"
    return "ОК"

def worker():
    while True:
        # Получаем событие из очереди
        content = event_queue.get()
        if content is None:
            # Завершаем работу потока, если получен сигнал остановки
            break
        result = process_content(content)
        # Отправляем результат в очередь ответов
        response_queue.put(result)
        # Сообщаем, что задача выполнена
        event_queue.task_done()

def start_processing(content):
    # Добавляем событие в очередь
    event_queue.put(content)

@app.route("/process", methods=["POST"])
def process_request():
    content = request.json.get("content")
    start_processing(content)
    return jsonify({"message": "Request received."})

if __name__ == "__main__":
    # Создаем и запускаем потоки-обработчики
    num_workers = 4  # Количество потоков
    workers = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)
    app.run()
    
    # Останавливаем потоки
    for _ in range(num_workers):
        event_queue.put(None)
    for t in workers:
        t.join()
