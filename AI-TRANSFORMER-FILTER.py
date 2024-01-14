from flask import Flask, request, jsonify
from transformers import GPT2LMTokenizer, GPT2LMModel
import torch
import threading
from queue import Queue

app = Flask(__name__)

# Загрузка предварительно обученной модели GPT-3
model_name = 'gpt2'
model = GPT2LMModel.from_pretrained(model_name)
tokenizer = GPT2LMTokenizer.from_pretrained(model_name)

# Очередь событий для хранения запросов
event_queue = Queue()

def generate_response(prompt):
    # Предварительная обработка текста
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Генерация ответа с помощью модели GPT-3
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def worker():
    while True:
        # Получаем событие из очереди
        prompt = event_queue.get()
        if prompt is None:
            # Завершаем работу потока, если получен сигнал остановки
            break
        response = generate_response(prompt)
        # Отправляем результат в очередь ответов
        response_queue.put(response)
        # Сообщаем, что задача выполнена
        event_queue.task_done()

def start_processing(prompt):
    # Добавляем событие в очередь
    event_queue.put(prompt)

@app.route("/generate", methods=["POST"])
def generate_request():
    prompt = request.json.get("prompt")
    start_processing(prompt)
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
