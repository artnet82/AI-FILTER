import torch
from transformers import AutoTokenizer, AutoModel

def check_forbidden_words(model_name, text, forbidden_words):
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Токенизация текста
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])

    # Применение модели к токенизированному тексту
    model_output = model(input_tensor)

    # Проверка наличия запрещенных слов
    for word in forbidden_words:
        if word in text:
            print(f"Обнаружено запрещенное слово: {word}")

def main():
    model_name = "model_name"  # Замените на имя или путь к вашей модели
    text = "Пример текста для проверки модели на запрещенные слова."
    forbidden_words = ["запрещенное", "слово", "недопустимое"]  # Список запрещенных слов

    check_forbidden_words(model_name, text, forbidden_words)

if __name__ == '__main__':
    main()
